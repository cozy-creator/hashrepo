package hashrepo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGrantMatchesSharedV1Vector(t *testing.T) {
	vector, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", "upload_grant.json"))
	if err != nil {
		t.Fatal(err)
	}
	vector = bytes.TrimSpace(vector)

	var grant Grant
	if err := json.Unmarshal(vector, &grant); err != nil {
		t.Fatalf("decode shared grant vector: %v", err)
	}
	wantDigest, err := ParseRef("sha256:0ad1863ee4d0195b751580cc2e1be191255b27964174273c2ace87fad35123c9")
	if err != nil {
		t.Fatal(err)
	}
	wantExpiry := time.Date(2026, 8, 13, 12, 10, 0, 0, time.UTC)
	if grant.Digest != wantDigest || grant.SizeBytes != 18 ||
		grant.StagingKey != "staging/sha256/session-1/"+wantDigest.Hex() ||
		grant.URL != "https://objects.invalid/upload?token=v1" ||
		grant.ExpiresAt != wantExpiry {
		t.Fatalf("decoded grant does not match v1 vector: %+v", grant)
	}
	if grant.Headers == nil || len(grant.Headers) != 0 {
		t.Fatalf("headers = %#v, want a non-nil empty map", grant.Headers)
	}

	// A store adapter is allowed to return nil for no headers. The public wire
	// contract still requires an empty JSON object rather than null.
	grant.Headers = nil
	encoded, err := json.Marshal(grant)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, vector) {
		t.Fatalf("grant wire mismatch\n got: %s\nwant: %s", encoded, vector)
	}
}

func TestGrantRefusesNullOrAbsentHeaders(t *testing.T) {
	for _, raw := range []string{
		`{"digest":"sha256:0ad1863ee4d0195b751580cc2e1be191255b27964174273c2ace87fad35123c9","size_bytes":18,"staging_key":"staging/sha256/session/x","url":"https://objects.invalid","headers":null,"expires_at":"2026-08-13T12:10:00Z"}`,
		`{"digest":"sha256:0ad1863ee4d0195b751580cc2e1be191255b27964174273c2ace87fad35123c9","size_bytes":18,"staging_key":"staging/sha256/session/x","url":"https://objects.invalid","expires_at":"2026-08-13T12:10:00Z"}`,
		`{"digest":"sha256:0ad1863ee4d0195b751580cc2e1be191255b27964174273c2ace87fad35123c9","size_bytes":18,"staging_key":"staging/sha256/session/x","url":"https://objects.invalid","headers":{},"expires_at":"2026-08-13T12:10:00+00:00"}`,
		`{"digest":"sha256:0ad1863ee4d0195b751580cc2e1be191255b27964174273c2ace87fad35123c9","size_bytes":18,"staging_key":"staging/sha256/session/x","url":"https://objects.invalid","headers":{},"expires_at":"2026-08-13T12:10:00Z","extra":true}`,
	} {
		var grant Grant
		if err := json.Unmarshal([]byte(raw), &grant); err == nil {
			t.Fatalf("accepted invalid grant: %s", raw)
		}
	}
}

type memoryStore struct {
	resident map[Ref]bool
	staged   map[Ref]bool
	granted  []StagedObject
	emptyURL bool
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
	if s.emptyURL {
		return "", nil, nil
	}
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

type unknownClaimGate struct{}

func (unknownClaimGate) Unclaimable(_ context.Context, _ []Ref) ([]Ref, error) {
	return []Ref{RefBytes([]byte("unknown"))}, nil
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

func TestPlanFailsClosedWhenClaimGateSubstitutesAnObject(t *testing.T) {
	manifest := sharedManifest(t)
	resident := map[Ref]bool{}
	for _, file := range manifest.Files {
		for _, object := range file.Objects() {
			resident[object.Digest] = true
		}
	}
	store := &memoryStore{resident: resident, staged: map[Ref]bool{}}
	_, err := (Planner{Store: store, Claims: unknownClaimGate{}}).Plan(
		context.Background(), "session", manifest,
	)
	if err == nil {
		t.Fatal("claim-gate substitution was accepted")
	}
}

func TestPlanRefusesEmptyGrantURL(t *testing.T) {
	store := &memoryStore{
		resident: map[Ref]bool{}, staged: map[Ref]bool{}, emptyURL: true,
	}
	_, err := (Planner{Store: store}).Plan(context.Background(), "session", sharedManifest(t))
	if err == nil {
		t.Fatal("empty grant URL was accepted")
	}
}

func TestValidateManifestEnforcesLimitsBeforeStoreAccess(t *testing.T) {
	manifest := sharedManifest(t)
	for name, limits := range map[string]Limits{
		"files":       {MaxFiles: len(manifest.Files) - 1},
		"objects":     {MaxObjects: 1},
		"file bytes":  {MaxFileBytes: 1},
		"total bytes": {MaxTotalBytes: 1},
	} {
		t.Run(name, func(t *testing.T) {
			store := &memoryStore{resident: map[Ref]bool{}, staged: map[Ref]bool{}}
			if _, err := (Planner{Store: store, Limits: limits}).Plan(
				context.Background(), "session", manifest,
			); err == nil {
				t.Fatal("bounded declaration was accepted")
			}
			if len(store.granted) != 0 {
				t.Fatal("store was reached before limit validation")
			}
		})
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
