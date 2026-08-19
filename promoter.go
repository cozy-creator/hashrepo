package tensorfs

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
)

const defaultPromotionParallel = 8

// StoredObject is an object store's answer about one exact key.
//
// Promotion requires a store-asserted SHA-256 checksum. Presence and byte
// length alone never prove that a content-addressed destination is correct.
type StoredObject struct {
	Present   bool
	SizeBytes int64
	Checksum  *Ref
}

// PromotionStore is the object-store seam required for publication.
//
// th#2184 — there is ONE call, and it moves no bytes. Uploads are granted
// straight at the content-addressed destination with the digest bound inside
// the signature, so by the time a promotion runs the store either holds the
// declared bytes at the declared key or it does not, and the promotion's whole
// job is to say which with the store's own assertion. The copy this interface
// used to demand made every promoted byte cross the process that minted the
// grant; deleting it is the fix, not an optimization of it.
//
// Implementations are called concurrently.
type PromotionStore interface {
	Inspect(context.Context, string) (StoredObject, error)
}

// PromotionReport is one retryable promotion pass with explicit denominators.
type PromotionReport struct {
	Objects           int      `json:"objects"`
	Resident          int      `json:"resident"`
	ChecksumConfirmed int      `json:"checksum_confirmed"`
	Failed            []Ref    `json:"failed,omitempty"`
	Errors            []string `json:"errors,omitempty"`
}

// Complete reports whether every planned destination was checksum-confirmed.
func (r PromotionReport) Complete() bool {
	return len(r.Failed) == 0 &&
		r.ChecksumConfirmed == r.Objects &&
		r.Resident == r.Objects
}

// Promoter confirms that every declared object is resident at its immutable
// CAS key under the store's own checksum assertion.
type Promoter struct {
	Store        PromotionStore
	ObjectPrefix string
	Parallel     int
}

type promotionResult struct {
	resident bool
	err      error
}

func requireStoredObject(state StoredObject, object PlannedObject, where string) error {
	if !state.Present {
		return fmt.Errorf("%s object is absent", where)
	}
	if state.SizeBytes != object.SizeBytes {
		return fmt.Errorf(
			"%s object is %d bytes, expected %d", where, state.SizeBytes, object.SizeBytes,
		)
	}
	if state.Checksum == nil {
		return fmt.Errorf("%s object has no store-asserted checksum", where)
	}
	if *state.Checksum != object.Digest {
		return fmt.Errorf("%s object checksum is %s, expected %s", where, state.Checksum, object.Digest)
	}
	return nil
}

func declaredObjects(manifest Manifest) (map[Ref]int64, error) {
	validated, err := ValidateManifest(manifest, Limits{})
	if err != nil {
		return nil, err
	}
	objects := map[Ref]int64{}
	for _, file := range validated.Files {
		for _, object := range file.Objects() {
			if previous, found := objects[object.Digest]; found && previous != object.SizeBytes {
				return nil, fmt.Errorf(
					"object %s declared with two sizes (%d and %d)",
					object.Digest, previous, object.SizeBytes,
				)
			}
			objects[object.Digest] = object.SizeBytes
		}
	}
	return objects, nil
}

// validatePromotionPlan turns the mutable wire Plan into a trusted work list.
//
// Every entry — resident-at-declare and granted-but-not-yet-observed alike —
// is adjudicated at the SAME key, recomputed here from the digest and the
// plan's own namespace. A caller-supplied key is never trusted and never
// followed, which is what keeps a mutated plan from pointing a confirmation at
// somebody else's object.
func validatePromotionPlan(plan Plan, prefix string) ([]PlannedObject, error) {
	if !sessionPattern.MatchString(plan.SessionID) {
		return nil, errors.New("promotion plan has an invalid session id")
	}
	declared, err := declaredObjects(plan.Manifest)
	if err != nil {
		return nil, fmt.Errorf("promotion manifest: %w", err)
	}
	partitioned := make(map[Ref]bool, len(declared))
	objects := make([]PlannedObject, 0, len(declared))
	add := func(ref Ref, size int64) error {
		if partitioned[ref] {
			return fmt.Errorf("object %s appears more than once in the plan", ref)
		}
		key, keyErr := UploadKey(prefix, ref)
		if keyErr != nil {
			return keyErr
		}
		partitioned[ref] = true
		objects = append(objects, PlannedObject{
			Object: Object{Digest: ref, SizeBytes: size}, UploadKey: key,
		})
		return nil
	}
	for _, ref := range plan.Have {
		size, found := declared[ref]
		if !found {
			return nil, fmt.Errorf("resident object %s is not declared by the manifest", ref)
		}
		if err := add(ref, size); err != nil {
			return nil, err
		}
	}
	for _, object := range plan.PendingObjects() {
		size, found := declared[object.Digest]
		if !found {
			return nil, fmt.Errorf("granted object %s is not declared by the manifest", object.Digest)
		}
		if size != object.SizeBytes {
			return nil, fmt.Errorf(
				"granted object %s is %d bytes, manifest declares %d",
				object.Digest, object.SizeBytes, size,
			)
		}
		expected, keyErr := UploadKey(prefix, object.Digest)
		if keyErr != nil {
			return nil, keyErr
		}
		if object.UploadKey != expected {
			return nil, fmt.Errorf(
				"grant key %q is outside the plan namespace; expected %q",
				object.UploadKey, expected,
			)
		}
		if err := add(object.Digest, size); err != nil {
			return nil, err
		}
	}
	if len(partitioned) != len(declared) {
		return nil, fmt.Errorf(
			"promotion plan partitions %d of %d declared objects", len(partitioned), len(declared),
		)
	}
	return objects, nil
}

func (p Promoter) promoteOne(ctx context.Context, object PlannedObject) promotionResult {
	if err := ctx.Err(); err != nil {
		return promotionResult{err: err}
	}
	state, err := p.Store.Inspect(ctx, object.UploadKey)
	if err != nil {
		return promotionResult{err: fmt.Errorf("residency inspection: %w", err)}
	}
	if !state.Present {
		// RETRYABLE and precise: the grant was minted, the bytes never
		// arrived. A later pass over the same content-addressed key confirms
		// whatever has landed since.
		return promotionResult{err: errors.New("object is absent at its content-addressed key")}
	}
	// A poisoned destination is repaired the same way it was written: the
	// planner reports it non-resident and grants a fresh checksum-enforced
	// upload over it. There is nothing for a promotion to heal.
	if err := requireStoredObject(state, object, "resident"); err != nil {
		return promotionResult{err: err}
	}
	return promotionResult{resident: true}
}

// Promote validates the complete plan before making any store call, then
// checksum-confirms every destination. It reads NO object bytes: the whole
// pass is one HEAD per distinct object. Per-object failures remain retryable —
// the destination is content-addressed, so a later pass sees whatever landed
// in between and nothing it confirmed can change underneath it.
func (p Promoter) Promote(ctx context.Context, plan Plan) (PromotionReport, error) {
	if p.Store == nil {
		return PromotionReport{}, errors.New("tensorfs: promotion store is required")
	}
	if p.ObjectPrefix == "" {
		p.ObjectPrefix = defaultObjectPrefix
	}
	// The plan's own namespace wins when it carries one: grants were minted
	// into it, and confirming a different one would look for objects nobody
	// was ever authorized to write.
	prefix := p.ObjectPrefix
	if planPrefix := strings.TrimSpace(plan.ObjectPrefix); planPrefix != "" {
		prefix = planPrefix
	}
	objects, err := validatePromotionPlan(plan, prefix)
	if err != nil {
		return PromotionReport{}, err
	}
	report := PromotionReport{Objects: len(objects)}
	if len(objects) == 0 {
		return report, nil
	}
	parallel := p.Parallel
	if parallel <= 0 {
		parallel = defaultPromotionParallel
	}
	parallel = min(parallel, len(objects))
	results := make([]promotionResult, len(objects))
	jobs := make(chan int)
	var workers sync.WaitGroup
	workers.Add(parallel)
	for range parallel {
		go func() {
			defer workers.Done()
			for index := range jobs {
				results[index] = p.promoteOne(ctx, objects[index])
			}
		}()
	}
	sent := 0
send:
	for index := range objects {
		select {
		case jobs <- index:
			sent++
		case <-ctx.Done():
			break send
		}
	}
	close(jobs)
	workers.Wait()
	for index := sent; index < len(objects); index++ {
		results[index].err = ctx.Err()
	}

	for index, result := range results {
		object := objects[index]
		switch {
		case result.err != nil:
			report.Failed = append(report.Failed, object.Digest)
			report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", object.Digest, result.err))
		case result.resident:
			report.Resident++
			report.ChecksumConfirmed++
		}
	}
	return report, nil
}
