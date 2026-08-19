package tensorfs

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

type promotionMemoryStore struct {
	objects        map[string]StoredObject
	inspectErrors  map[string]error
	inspectionKeys []string
	mu             sync.Mutex
}

func (s *promotionMemoryStore) Inspect(_ context.Context, key string) (StoredObject, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inspectionKeys = append(s.inspectionKeys, key)
	if err := s.inspectErrors[key]; err != nil {
		return StoredObject{}, err
	}
	return s.objects[key], nil
}

// promotionPlan is the ordinary post-upload shape: one object the planner found
// resident at declare time, one it granted. Under th#2184 both live at the SAME
// final key, and the promotion tells them apart only by where they came from in
// the plan — which is exactly why it does the same thing to both.
func promotionPlan(t *testing.T) (Plan, PlannedObject, PlannedObject) {
	t.Helper()
	key := func(ref Ref) string {
		out, err := ref.ObjectKey("objects")
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	first := PlannedObject{
		Object:    Object{Digest: RefBytes([]byte("first")), SizeBytes: 5},
		UploadKey: key(RefBytes([]byte("first"))),
	}
	second := PlannedObject{
		Object:    Object{Digest: RefBytes([]byte("second")), SizeBytes: 6},
		UploadKey: key(RefBytes([]byte("second"))),
	}
	return Plan{
		SessionID: "session",
		Manifest: Manifest{Format: FormatV1, Files: []File{
			{Path: "first", SizeBytes: 5, Digest: first.Digest},
			{Path: "second", SizeBytes: 6, Digest: second.Digest},
		}},
		Have: []Ref{first.Digest},
		Need: []Grant{{PlannedObject: second, URL: "https://objects.invalid/second"}},
	}, first, second
}

func promotionStore(first, second PlannedObject) *promotionMemoryStore {
	firstChecksum, secondChecksum := first.Digest, second.Digest
	return &promotionMemoryStore{
		objects: map[string]StoredObject{
			first.UploadKey:  {Present: true, SizeBytes: 5, Checksum: &firstChecksum},
			second.UploadKey: {Present: true, SizeBytes: 6, Checksum: &secondChecksum},
		},
		inspectErrors: map[string]error{},
	}
}

func TestPromoteChecksumConfirmsEveryDestinationAndMovesNoBytes(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	report, err := (Promoter{Store: store, Parallel: 2}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Complete() || report.Resident != 2 || report.ChecksumConfirmed != 2 {
		t.Fatalf("unexpected report: %+v", report)
	}
	// THE POINT OF THE ISSUE: one HEAD per object and nothing else. There is no
	// second seam on PromotionStore that could read or write a byte.
	if len(store.inspectionKeys) != 2 {
		t.Fatalf("inspections = %v, want exactly one per object", store.inspectionKeys)
	}
}

// TestPromoteConfirmsTheKeyItRECOMPUTES is the fence that replaces the old
// session-prefix check: a plan is untrusted data, so the destination comes from
// the digest, never from the plan.
func TestPromoteConfirmsTheKeyItRecomputesNotTheOneThePlanCarries(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Need[0].UploadKey = "objects/unrelated"
	store := promotionStore(first, second)
	_, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err == nil {
		t.Fatal("a grant key outside the plan namespace was accepted")
	}
	if !strings.Contains(err.Error(), "outside the plan namespace") {
		t.Fatalf("unexpected refusal: %v", err)
	}
	if len(store.inspectionKeys) != 0 {
		t.Fatalf("store was accessed before plan validation: %+v", store.inspectionKeys)
	}
}

// TestPromoteHonoursThePlansOwnNamespace: grants were minted into the plan's
// prefix, so a Promoter configured with a different one must not go looking in
// its own.
func TestPromoteHonoursThePlansOwnNamespace(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.ObjectPrefix = "blobs"
	blobKey := func(ref Ref) string {
		out, err := ref.ObjectKey("blobs")
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	plan.Need[0].UploadKey = blobKey(second.Digest)
	firstChecksum, secondChecksum := first.Digest, second.Digest
	store := &promotionMemoryStore{
		objects: map[string]StoredObject{
			blobKey(first.Digest):  {Present: true, SizeBytes: 5, Checksum: &firstChecksum},
			blobKey(second.Digest): {Present: true, SizeBytes: 6, Checksum: &secondChecksum},
		},
		inspectErrors: map[string]error{},
	}
	report, err := (Promoter{Store: store, ObjectPrefix: "objects"}).Promote(context.Background(), plan)
	if err != nil || !report.Complete() {
		t.Fatalf("report = %+v, err = %v", report, err)
	}
	for _, key := range store.inspectionKeys {
		if !strings.HasPrefix(key, "blobs/") {
			t.Fatalf("inspected %q, want the plan's own namespace", key)
		}
	}
}

// TestPromoteReportsAnAbsentGrantAsRetryableNotAsSuccess — the upload never
// landed. Before th#2184 this was "the copy failed"; now it is "the bytes are
// not there", which is the same verdict for the client and one fewer moving
// part for the hub.
func TestPromoteReportsAnAbsentObjectAsRetryable(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	delete(store.objects, second.UploadKey)
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete() || len(report.Failed) != 1 || report.Failed[0] != second.Digest {
		t.Fatalf("unexpected report: %+v", report)
	}
	if len(report.Errors) != 1 || !strings.Contains(report.Errors[0], "absent at its content-addressed key") {
		t.Fatalf("errors = %v", report.Errors)
	}

	// The retry is the whole contract: put the object there and re-run the SAME
	// plan.
	checksum := second.Digest
	store.objects[second.UploadKey] = StoredObject{Present: true, SizeBytes: 6, Checksum: &checksum}
	retry, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil || !retry.Complete() {
		t.Fatalf("retry = %+v, err = %v", retry, err)
	}
}

func TestPromoteNeverCallsUnassertedOrShortDestinationComplete(t *testing.T) {
	for _, tc := range []struct {
		name  string
		state StoredObject
	}{
		{"no store-asserted checksum", StoredObject{Present: true, SizeBytes: 6}},
		{"wrong length", func() StoredObject {
			c := RefBytes([]byte("second"))
			return StoredObject{Present: true, SizeBytes: 5, Checksum: &c}
		}()},
		{"wrong digest", func() StoredObject {
			c := RefBytes([]byte("first"))
			return StoredObject{Present: true, SizeBytes: 6, Checksum: &c}
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, first, second := promotionPlan(t)
			store := promotionStore(first, second)
			store.objects[second.UploadKey] = tc.state
			report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
			if err != nil {
				t.Fatal(err)
			}
			if report.Complete() || report.ChecksumConfirmed != 1 || len(report.Failed) != 1 {
				t.Fatalf("unexpected report: %+v", report)
			}
		})
	}
}

func TestPromoteSurfacesAStoreErrorPerObjectAndStaysRetryable(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	store.inspectErrors[second.UploadKey] = errors.New("transient")
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete() || len(report.Failed) != 1 || !strings.Contains(report.Errors[0], "transient") {
		t.Fatalf("unexpected report: %+v", report)
	}
}

func TestPromoteRefusesIncompleteManifestPartition(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Need = nil
	store := promotionStore(first, second)
	_, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err == nil {
		t.Fatal("incomplete plan was accepted")
	}
	if len(store.inspectionKeys) != 0 {
		t.Fatal("store was accessed for an incomplete plan")
	}
}

func TestPromoteRefusesADuplicatedObject(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Have = []Ref{first.Digest, second.Digest}
	store := promotionStore(first, second)
	_, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err == nil {
		t.Fatal("an object planned twice was accepted")
	}
}
