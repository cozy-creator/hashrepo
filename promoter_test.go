package hashrepo

import (
	"context"
	"errors"
	"sync"
	"testing"
)

type promotionMemoryStore struct {
	objects        map[string]StoredObject
	promoteErrors  map[string]error
	promotions     [][2]string
	dropChecksum   bool
	inspectionKeys []string
	mu             sync.Mutex
}

func (s *promotionMemoryStore) Inspect(_ context.Context, key string) (StoredObject, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inspectionKeys = append(s.inspectionKeys, key)
	return s.objects[key], nil
}

func (s *promotionMemoryStore) PromoteVerified(
	_ context.Context, source StagedObject, destination string,
) (StoredObject, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.promoteErrors[source.StagingKey]; err != nil {
		return StoredObject{}, err
	}
	staged := s.objects[source.StagingKey]
	if err := requireStoredObject(staged, source, "staged"); err != nil {
		return StoredObject{}, err
	}
	resident := staged
	if s.dropChecksum {
		resident.Checksum = nil
	}
	s.promotions = append(s.promotions, [2]string{source.StagingKey, destination})
	s.objects[destination] = resident
	return resident, nil
}

func promotionPlan(t *testing.T) (Plan, StagedObject, StagedObject) {
	t.Helper()
	first := StagedObject{
		Object:     Object{Digest: RefBytes([]byte("first")), SizeBytes: 5},
		StagingKey: "staging/sha256/session/" + RefBytes([]byte("first")).Hex(),
	}
	second := StagedObject{
		Object:     Object{Digest: RefBytes([]byte("second")), SizeBytes: 6},
		StagingKey: "staging/sha256/session/" + RefBytes([]byte("second")).Hex(),
	}
	return Plan{
		SessionID: "session",
		Manifest: Manifest{Format: FormatV1, Files: []File{
			{Path: "first", SizeBytes: 5, Digest: first.Digest},
			{Path: "second", SizeBytes: 6, Digest: second.Digest},
		}},
		Staged: []StagedObject{first},
		Need: []Grant{{
			StagedObject: second,
			URL:          "https://objects.invalid/second",
		}},
	}, first, second
}

func promotionStore(first, second StagedObject) *promotionMemoryStore {
	firstChecksum, secondChecksum := first.Digest, second.Digest
	return &promotionMemoryStore{
		objects: map[string]StoredObject{
			first.StagingKey:  {Present: true, SizeBytes: 5, Checksum: &firstChecksum},
			second.StagingKey: {Present: true, SizeBytes: 6, Checksum: &secondChecksum},
		},
		promoteErrors: map[string]error{},
	}
}

func TestPromoteAtomicallyVerifiesAndChecksumConfirmsDestinations(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	report, err := (Promoter{Store: store, Parallel: 2}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Complete() || report.Copied != 2 || report.ChecksumConfirmed != 2 {
		t.Fatalf("unexpected report: %+v", report)
	}
}

func TestPromoteRefusesUnverifiedStagingAndIsRetryable(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	store.objects[first.StagingKey] = StoredObject{Present: true, SizeBytes: 5}
	store.promoteErrors[second.StagingKey] = errors.New("transient")
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete() || len(report.Failed) != 2 || len(store.promotions) != 0 {
		t.Fatalf("unexpected report: %+v, promotions=%v", report, store.promotions)
	}

	firstChecksum := first.Digest
	store.objects[first.StagingKey] = StoredObject{Present: true, SizeBytes: 5, Checksum: &firstChecksum}
	delete(store.promoteErrors, second.StagingKey)
	retry, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil || !retry.Complete() {
		t.Fatalf("retry = %+v, err=%v", retry, err)
	}
}

func TestPromoteRecognizesChecksumConfirmedResidentRetry(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Have = []Ref{second.Digest}
	plan.Need = nil
	store := promotionStore(first, second)
	checksum := first.Digest
	destination, err := first.Digest.ObjectKey("objects")
	if err != nil {
		t.Fatal(err)
	}
	store.objects[destination] = StoredObject{Present: true, SizeBytes: 5, Checksum: &checksum}
	secondChecksum := second.Digest
	secondDestination, err := second.Digest.ObjectKey("objects")
	if err != nil {
		t.Fatal(err)
	}
	store.objects[secondDestination] = StoredObject{
		Present: true, SizeBytes: 6, Checksum: &secondChecksum,
	}
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil || !report.Complete() || report.AlreadyResident != 2 || report.Objects != 2 {
		t.Fatalf("report = %+v, err=%v", report, err)
	}
}

func TestPromoteAuditsEveryHaveDestination(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Have = []Ref{second.Digest}
	plan.Need = nil
	store := promotionStore(first, second)
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete() || report.Objects != 2 || len(report.Failed) != 1 ||
		report.Failed[0] != second.Digest {
		t.Fatalf("missing Have destination was accepted: %+v", report)
	}
}

func TestPromoteRefusesArbitraryStagingKeyBeforeStoreAccess(t *testing.T) {
	plan, first, second := promotionPlan(t)
	plan.Staged[0].StagingKey = "objects/unrelated"
	store := promotionStore(first, second)
	_, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err == nil {
		t.Fatal("arbitrary staging key was accepted")
	}
	if len(store.inspectionKeys) != 0 || len(store.promotions) != 0 {
		t.Fatalf("store was accessed before plan validation: %+v", store)
	}
}

func TestPromoteNeverCallsUnassertedDestinationComplete(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	store.dropChecksum = true
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete() || len(report.Failed) != 2 || report.ChecksumConfirmed != 0 {
		t.Fatalf("unexpected report: %+v", report)
	}
}

func TestPromoteRepairsPoisonedDestinationThroughVerifiedPromotion(t *testing.T) {
	plan, first, second := promotionPlan(t)
	store := promotionStore(first, second)
	destination, err := first.Digest.ObjectKey("objects")
	if err != nil {
		t.Fatal(err)
	}
	wrong := second.Digest
	store.objects[destination] = StoredObject{Present: true, SizeBytes: 5, Checksum: &wrong}
	report, err := (Promoter{Store: store}).Promote(context.Background(), plan)
	if err != nil || !report.Complete() || report.Copied != 2 {
		t.Fatalf("report = %+v, err=%v", report, err)
	}
	if stored := store.objects[destination]; stored.Checksum == nil || *stored.Checksum != first.Digest {
		t.Fatalf("poisoned destination was not repaired: %+v", stored)
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
