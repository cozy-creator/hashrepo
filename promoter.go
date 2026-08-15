package tensorfs

import (
	"context"
	"errors"
	"fmt"
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

// PromotionStore is the object-store seam required for staged publication.
//
// Implementations are called concurrently. PromoteVerified must atomically
// bind verification of the staged source's digest and length to the copy, then
// return the store-asserted state of the immutable destination. A plain
// inspect-then-copy sequence does not satisfy this contract.
type PromotionStore interface {
	Inspect(context.Context, string) (StoredObject, error)
	PromoteVerified(context.Context, StagedObject, string) (StoredObject, error)
}

// PromotionReport is one retryable promotion pass with explicit denominators.
type PromotionReport struct {
	Objects           int      `json:"objects"`
	Copied            int      `json:"copied"`
	AlreadyResident   int      `json:"already_resident"`
	ChecksumConfirmed int      `json:"checksum_confirmed"`
	Failed            []Ref    `json:"failed,omitempty"`
	Errors            []string `json:"errors,omitempty"`
}

// Complete reports whether every planned destination was checksum-confirmed.
func (r PromotionReport) Complete() bool {
	return len(r.Failed) == 0 &&
		r.ChecksumConfirmed == r.Objects &&
		r.Copied+r.AlreadyResident == r.Objects
}

// Promoter verifies and promotes staged uploads into immutable CAS keys.
type Promoter struct {
	Store        PromotionStore
	ObjectPrefix string
	Parallel     int
}

type promotionResult struct {
	copied          bool
	alreadyResident bool
	err             error
}

type promotionObject struct {
	StagedObject
	residentOnly bool
}

func requireStoredObject(state StoredObject, object StagedObject, where string) error {
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
// It also proves that staging cleanup can be left exclusively to the
// session-scoped object-store lifecycle; the generic promoter never deletes a
// caller-supplied key.
func validatePromotionPlan(plan Plan) ([]promotionObject, error) {
	if !sessionPattern.MatchString(plan.SessionID) {
		return nil, errors.New("promotion plan has an invalid session id")
	}
	declared, err := declaredObjects(plan.Manifest)
	if err != nil {
		return nil, fmt.Errorf("promotion manifest: %w", err)
	}
	partitioned := make(map[Ref]bool, len(declared))
	objects := make([]promotionObject, 0, len(declared))
	for _, ref := range plan.Have {
		size, found := declared[ref]
		if !found {
			return nil, fmt.Errorf("resident object %s is not declared by the manifest", ref)
		}
		if partitioned[ref] {
			return nil, fmt.Errorf("object %s appears more than once in the plan", ref)
		}
		partitioned[ref] = true
		objects = append(objects, promotionObject{
			StagedObject: StagedObject{Object: Object{Digest: ref, SizeBytes: size}},
			residentOnly: true,
		})
	}
	pending := plan.PendingObjects()
	for _, object := range pending {
		size, found := declared[object.Digest]
		if !found {
			return nil, fmt.Errorf("staged object %s is not declared by the manifest", object.Digest)
		}
		if size != object.SizeBytes {
			return nil, fmt.Errorf(
				"staged object %s is %d bytes, manifest declares %d",
				object.Digest, object.SizeBytes, size,
			)
		}
		if partitioned[object.Digest] {
			return nil, fmt.Errorf("object %s appears more than once in the plan", object.Digest)
		}
		expected, keyErr := StagedKey(plan.SessionID, object.Digest)
		if keyErr != nil {
			return nil, keyErr
		}
		if object.StagingKey != expected {
			return nil, fmt.Errorf(
				"staging key %q is outside session %q; expected %q",
				object.StagingKey, plan.SessionID, expected,
			)
		}
		partitioned[object.Digest] = true
		objects = append(objects, promotionObject{StagedObject: object})
	}
	if len(partitioned) != len(declared) {
		return nil, fmt.Errorf(
			"promotion plan partitions %d of %d declared objects", len(partitioned), len(declared),
		)
	}
	return objects, nil
}

func (p Promoter) promoteOne(ctx context.Context, planned promotionObject) promotionResult {
	object := planned.StagedObject
	if err := ctx.Err(); err != nil {
		return promotionResult{err: err}
	}
	destination, err := object.Digest.ObjectKey(p.ObjectPrefix)
	if err != nil {
		return promotionResult{err: err}
	}
	resident, err := p.Store.Inspect(ctx, destination)
	if err != nil {
		return promotionResult{err: fmt.Errorf("resident inspection: %w", err)}
	}
	if resident.Present {
		if err := requireStoredObject(resident, object, "resident"); err == nil {
			return promotionResult{alreadyResident: true}
		}
		// A poisoned destination can be repaired only through the same atomic,
		// verified staging operation used for first publication.
	}
	if planned.residentOnly {
		if resident.Present {
			return promotionResult{err: errors.New("resident object has invalid checksum or size")}
		}
		return promotionResult{err: errors.New("resident object is absent")}
	}

	resident, err = p.Store.PromoteVerified(ctx, object, destination)
	if err != nil {
		return promotionResult{err: fmt.Errorf("verified promotion: %w", err)}
	}
	if err := requireStoredObject(resident, object, "promoted"); err != nil {
		return promotionResult{err: err}
	}
	return promotionResult{copied: true}
}

// Promote validates the complete plan before making any store call, then
// checksum-confirms every destination. Per-object failures remain retryable.
func (p Promoter) Promote(ctx context.Context, plan Plan) (PromotionReport, error) {
	if p.Store == nil {
		return PromotionReport{}, errors.New("tensorfs: promotion store is required")
	}
	objects, err := validatePromotionPlan(plan)
	if err != nil {
		return PromotionReport{}, err
	}
	if p.ObjectPrefix == "" {
		p.ObjectPrefix = "objects"
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
		object := objects[index].StagedObject
		switch {
		case result.err != nil:
			report.Failed = append(report.Failed, object.Digest)
			report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", object.Digest, result.err))
		case result.copied:
			report.Copied++
			report.ChecksumConfirmed++
		case result.alreadyResident:
			report.AlreadyResident++
			report.ChecksumConfirmed++
		}
	}
	return report, nil
}
