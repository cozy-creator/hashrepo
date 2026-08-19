package tensorfs

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
		grant.UploadKey != "blobs/sha256/"+wantDigest.Hex()[:2]+"/"+wantDigest.Hex()[2:4]+"/"+wantDigest.Hex() ||
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

func TestEverySharedInvalidGrantIsRefused(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", "invalid_upload_grants.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Name  string          `json:"name"`
		Grant json.RawMessage `json:"grant"`
	}
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatal(err)
	}
	for _, vector := range vectors {
		t.Run(vector.Name, func(t *testing.T) {
			var grant Grant
			if err := json.Unmarshal(vector.Grant, &grant); err == nil {
				t.Fatal("invalid grant was accepted")
			}
		})
	}
}

type memoryStore struct {
	resident map[Ref]bool
	staged   map[Ref]bool
	granted  []PlannedObject
	sessions []string
	emptyURL bool
}

func (s *memoryStore) Residency(_ context.Context, objects []Object) (map[Ref]bool, error) {
	answer := make(map[Ref]bool, len(objects))
	for _, object := range objects {
		answer[object.Digest] = s.resident[object.Digest]
	}
	return answer, nil
}

func (s *memoryStore) PresignPut(_ context.Context, sessionID string, object PlannedObject, _ time.Duration) (string, map[string]string, error) {
	s.granted = append(s.granted, object)
	s.sessions = append(s.sessions, sessionID)
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

func TestPlanUsesManifestLengthsForVariableChunks(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", "variable_manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := ParseManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	resident := map[Ref]bool{}
	for _, file := range manifest.Files {
		for _, object := range file.Objects() {
			resident[object.Digest] = true
		}
	}
	plan, err := (Planner{Store: &memoryStore{
		resident: resident,
		staged:   map[Ref]bool{},
	}}).Plan(context.Background(), "variable-layout", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if plan.DeclaredBytes != 27 || plan.DistinctObjects != 3 || len(plan.Have) != 3 {
		t.Fatalf("planner ignored explicit chunk lengths: %+v", plan)
	}
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
		Store:        store,
		Claims:       denyClaims{hello: true},
		ObjectPrefix: "blobs",
		GrantTTL:     10 * time.Minute,
		Now:          func() time.Time { return now },
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
	// th#2184: a previously-uploaded object is RESIDENT — it sits at its final
	// key — so there is no second "staged" bucket and no second probe. The
	// store's `staged` map is deliberately populated here and deliberately
	// ignored: `firstChunk` is granted again because nothing observed it.
	if len(plan.Need) != 3 {
		t.Fatalf("need = %d, want both chunks plus unclaimable hello", len(plan.Need))
	}
	if plan.Need[0].ExpiresAt != now.Add(10*time.Minute) {
		t.Fatalf("grant expiry = %s", plan.Need[0].ExpiresAt)
	}
	if len(plan.PendingObjects()) != 3 {
		t.Fatalf("pending = %d, want the granted set", len(plan.PendingObjects()))
	}
	// Every grant points at the FINAL content key, and the session id reaches
	// the store so it can bind a possession witness into the same signature.
	for _, grant := range plan.Need {
		want, err := grant.Digest.ObjectKey("blobs")
		if err != nil {
			t.Fatal(err)
		}
		if grant.UploadKey != want {
			t.Fatalf("grant key = %q, want the final CAS key %q", grant.UploadKey, want)
		}
	}
	for _, session := range store.sessions {
		if session != "session-1" {
			t.Fatalf("PresignPut saw session %q, want session-1", session)
		}
	}
}

// TestPlanReplanAdoptsWhatLandedWithoutASecondProbe is the resume proof: the
// bytes a dead pod already PUT are found by the ORDINARY residency probe,
// because they are already at their final key.
func TestPlanReplanAdoptsWhatLandedWithoutASecondProbe(t *testing.T) {
	manifest := sharedManifest(t)
	store := &memoryStore{resident: map[Ref]bool{}, staged: map[Ref]bool{}}
	planner := Planner{Store: store, ObjectPrefix: "blobs"}
	first, err := planner.Plan(context.Background(), "session-1", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Need) != 4 || len(first.Have) != 0 {
		t.Fatalf("first plan = %d need / %d have, want 4/0", len(first.Need), len(first.Have))
	}
	// The client uploads three of the four. Nothing tells the hub; the objects
	// simply exist at their final keys now.
	for _, grant := range first.Need[:3] {
		store.resident[grant.Digest] = true
	}
	again, err := planner.Plan(context.Background(), "session-1", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Have) != 3 || len(again.Need) != 1 {
		t.Fatalf("re-plan = %d have / %d need, want 3/1", len(again.Have), len(again.Need))
	}
	// A DIFFERENT session resumes onto the same landed bytes — the old
	// session-scoped staging prefix made that impossible.
	crossPod, err := planner.Plan(context.Background(), "session-2", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(crossPod.Have) != 3 || len(crossPod.Need) != 1 {
		t.Fatalf("cross-session re-plan = %d have / %d need, want 3/1", len(crossPod.Have), len(crossPod.Need))
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
