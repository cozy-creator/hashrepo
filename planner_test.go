package chunkedcas

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type memoryStore struct {
	resident map[Ref]bool
	staged   map[Ref]bool
	granted  []StagedObject
}

func (s *memoryStore) Residency(_ context.Context, objects []Object) (map[Ref]bool, error) {
	answer := make(map[Ref]bool, len(objects))
	for _, object := range objects {
		answer[object.Digest] = s.resident[object.Digest]
	}
	return answer, nil
}

func (s *memoryStore) StagedResidency(_ context.Context, _ string, objects []Object) (map[Ref]bool, error) {
	answer := make(map[Ref]bool, len(objects))
	for _, object := range objects {
		answer[object.Digest] = s.staged[object.Digest]
	}
	return answer, nil
}

func (s *memoryStore) PresignPut(_ context.Context, object StagedObject, _ time.Duration) (string, map[string]string, error) {
	s.granted = append(s.granted, object)
	return "https://objects.invalid/" + object.Digest.Hex(), map[string]string{"x-test": "v1"}, nil
}

type denyClaims map[Ref]bool

func (d denyClaims) Unclaimable(_ context.Context, refs []Ref) ([]Ref, error) {
	var denied []Ref
	for _, ref := range refs {
		if d[ref] {
			denied = append(denied, ref)
		}
	}
	return denied, nil
}

func sharedManifest(t *testing.T) Manifest {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := ParseManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

func TestPlanDeduplicatesResumesAndHidesUnclaimableResidency(t *testing.T) {
	manifest := sharedManifest(t)
	empty := manifest.Files[0].Digest
	firstChunk := manifest.Files[1].Chunks[0].Digest
	hello := manifest.Files[2].Digest
	store := &memoryStore{
		resident: map[Ref]bool{empty: true, hello: true},
		staged:   map[Ref]bool{firstChunk: true},
	}
	now := time.Date(2026, 8, 13, 12, 0, 0, 0, time.UTC)
	planner := Planner{
		Store:    store,
		Claims:   denyClaims{hello: true},
		GrantTTL: 10 * time.Minute,
		Now:      func() time.Time { return now },
	}
	plan, err := planner.Plan(context.Background(), "session-1", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if plan.DistinctObjects != 4 || plan.ExaminedObjects != 4 {
		t.Fatalf("bad denominators: %+v", plan)
	}
	if len(plan.Have) != 1 || plan.Have[0] != empty {
		t.Fatalf("have = %v, want only empty object", plan.Have)
	}
	if len(plan.Staged) != 1 || plan.Staged[0].Digest != firstChunk {
		t.Fatalf("staged = %v, want first large-file chunk", plan.Staged)
	}
	if len(plan.Need) != 2 {
		t.Fatalf("need = %d, want final chunk plus unclaimable hello", len(plan.Need))
	}
	if plan.Need[0].ExpiresAt != now.Add(10*time.Minute) {
		t.Fatalf("grant expiry = %s", plan.Need[0].ExpiresAt)
	}
	if len(plan.PendingObjects()) != 3 {
		t.Fatalf("pending = %d, want staged + need", len(plan.PendingObjects()))
	}
}

func TestPlanFailsClosedOnPartialResidencyAnswer(t *testing.T) {
	store := &partialStore{memoryStore: memoryStore{resident: map[Ref]bool{}, staged: map[Ref]bool{}}}
	_, err := (Planner{Store: store}).Plan(context.Background(), "session", sharedManifest(t))
	if err == nil {
		t.Fatal("partial residency answer was accepted")
	}
}

func TestPlanFailsClosedWhenResidencySubstitutesAnUnknownObject(t *testing.T) {
	store := &substitutionStore{memoryStore: memoryStore{resident: map[Ref]bool{}, staged: map[Ref]bool{}}}
	_, err := (Planner{Store: store}).Plan(context.Background(), "session", sharedManifest(t))
	if err == nil {
		t.Fatal("substituted residency answer was accepted")
	}
}

type partialStore struct{ memoryStore }

type substitutionStore struct{ memoryStore }

func (s *partialStore) Residency(_ context.Context, objects []Object) (map[Ref]bool, error) {
	if len(objects) == 0 {
		return nil, fmt.Errorf("test requires objects")
	}
	return map[Ref]bool{objects[0].Digest: false}, nil
}

func (s *substitutionStore) Residency(_ context.Context, objects []Object) (map[Ref]bool, error) {
	answer := make(map[Ref]bool, len(objects))
	for _, object := range objects[1:] {
		answer[object.Digest] = false
	}
	answer[RefBytes([]byte("not declared"))] = false
	return answer, nil
}
