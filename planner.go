package tensorfs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path"
	"regexp"
	"sort"
	"strings"
	"time"
)

const defaultGrantTTL = 30 * time.Minute

const (
	defaultMaxFiles      = 100_000
	defaultMaxObjects    = 200_000
	defaultMaxFileBytes  = int64(512) << 30
	defaultMaxTotalBytes = int64(4) << 40
)

var sessionPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

// Object is one immutable digest and its exact byte length.
type Object struct {
	Digest    Ref   `json:"digest"`
	SizeBytes int64 `json:"size_bytes"`
}

// StagedObject is an object owned by one incomplete publication session.
type StagedObject struct {
	Object
	StagingKey string `json:"staging_key"`
}

// Grant is a verbatim, expiring HTTP PUT authorization.
type Grant struct {
	StagedObject
	URL       string            `json:"url"`
	Headers   map[string]string `json:"headers"`
	ExpiresAt time.Time         `json:"expires_at"`
}

type wireGrant struct {
	Digest     Ref               `json:"digest"`
	SizeBytes  *int64            `json:"size_bytes"`
	StagingKey string            `json:"staging_key"`
	URL        string            `json:"url"`
	Headers    map[string]string `json:"headers"`
	ExpiresAt  string            `json:"expires_at"`
}

// MarshalJSON keeps the v1 grant wire shape stable. In particular, no headers
// is an empty object rather than null: clients can always pass the value
// directly to an HTTP request without a nullable special case.
func (g Grant) MarshalJSON() ([]byte, error) {
	if g.Digest.hex == "" || g.SizeBytes < 0 || strings.TrimSpace(g.StagingKey) == "" ||
		strings.TrimSpace(g.URL) == "" || g.ExpiresAt.IsZero() {
		return nil, errors.New("grant requires digest, non-negative size, staging key, URL and expiry")
	}
	headers := g.Headers
	if headers == nil {
		headers = map[string]string{}
	}
	sizeBytes := g.SizeBytes
	return json.Marshal(wireGrant{
		Digest:     g.Digest,
		SizeBytes:  &sizeBytes,
		StagingKey: g.StagingKey,
		URL:        g.URL,
		Headers:    headers,
		ExpiresAt:  g.ExpiresAt.UTC().Format(time.RFC3339Nano),
	})
}

// UnmarshalJSON refuses nullable headers so readers enforce the same v1 shape
// writers emit.
func (g *Grant) UnmarshalJSON(data []byte) error {
	var wire wireGrant
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("grant must contain exactly one JSON value")
	}
	if wire.Headers == nil {
		return errors.New("grant headers must be an object, not null or absent")
	}
	expiresAt, err := time.Parse(time.RFC3339Nano, wire.ExpiresAt)
	if err != nil || !strings.HasSuffix(wire.ExpiresAt, "Z") {
		return errors.New("grant expiry must be an RFC 3339 UTC timestamp ending in Z")
	}
	if wire.Digest.hex == "" || wire.SizeBytes == nil || *wire.SizeBytes < 0 || strings.TrimSpace(wire.StagingKey) == "" ||
		strings.TrimSpace(wire.URL) == "" || expiresAt.IsZero() {
		return errors.New("grant requires digest, non-negative size, staging key, URL and expiry")
	}
	*g = Grant{
		StagedObject: StagedObject{
			Object:     Object{Digest: wire.Digest, SizeBytes: *wire.SizeBytes},
			StagingKey: wire.StagingKey,
		},
		URL:       wire.URL,
		Headers:   wire.Headers,
		ExpiresAt: expiresAt,
	}
	return nil
}

// Plan is the complete answer to a repository declaration.
type Plan struct {
	SessionID string         `json:"session_id"`
	Manifest  Manifest       `json:"manifest"`
	Have      []Ref          `json:"have"`
	Staged    []StagedObject `json:"staged,omitempty"`
	Need      []Grant        `json:"need"`

	DeclaredFiles   int   `json:"declared_files"`
	DeclaredBytes   int64 `json:"declared_bytes"`
	DistinctObjects int   `json:"distinct_objects"`
	ExaminedObjects int   `json:"examined_objects"`
	ResidentObjects int   `json:"resident_objects"`
	StagedObjects   int   `json:"staged_objects"`
}

// PendingObjects returns everything this session owns in staging, including
// objects found during a resumed declaration and newly granted objects.
func (p Plan) PendingObjects() []StagedObject {
	objects := append([]StagedObject(nil), p.Staged...)
	for _, grant := range p.Need {
		objects = append(objects, grant.StagedObject)
	}
	return objects
}

// Store is the narrow object-store seam used by the generic planner.
type Store interface {
	Residency(context.Context, []Object) (map[Ref]bool, error)
	StagedResidency(context.Context, string, []Object) (map[Ref]bool, error)
	PresignPut(context.Context, StagedObject, time.Duration) (string, map[string]string, error)
}

// ClaimGate hides resident objects a publisher is not authorized to claim.
// Such objects are planned as uploads, not disclosed as resident.
type ClaimGate interface {
	Unclaimable(context.Context, []Ref) ([]Ref, error)
}

// Limits bounds the work and bytes admitted by one declaration. Zero fields
// use conservative v1 defaults; a service may choose smaller policy limits.
type Limits struct {
	MaxFiles      int
	MaxObjects    int
	MaxFileBytes  int64
	MaxTotalBytes int64
}

func (limits Limits) defaults() Limits {
	if limits.MaxFiles <= 0 {
		limits.MaxFiles = defaultMaxFiles
	}
	if limits.MaxObjects <= 0 {
		limits.MaxObjects = defaultMaxObjects
	}
	if limits.MaxFileBytes <= 0 {
		limits.MaxFileBytes = defaultMaxFileBytes
	}
	if limits.MaxTotalBytes <= 0 {
		limits.MaxTotalBytes = defaultMaxTotalBytes
	}
	return limits
}

// ValidateManifest canonicalizes a v1 declaration and enforces resource
// bounds before a planner reaches an object store.
func ValidateManifest(manifest Manifest, limits Limits) (Manifest, error) {
	canonical, err := manifest.Canonical()
	if err != nil {
		return Manifest{}, err
	}
	manifest, err = ParseManifest(canonical)
	if err != nil {
		return Manifest{}, err
	}
	limits = limits.defaults()
	if len(manifest.Files) > limits.MaxFiles {
		return Manifest{}, fmt.Errorf(
			"manifest has %d files, limit is %d", len(manifest.Files), limits.MaxFiles,
		)
	}
	objects := map[Ref]int64{}
	var total int64
	for _, file := range manifest.Files {
		if file.SizeBytes > limits.MaxFileBytes {
			return Manifest{}, fmt.Errorf(
				"%s is %d bytes, per-file limit is %d",
				file.Path, file.SizeBytes, limits.MaxFileBytes,
			)
		}
		total, err = checkedAdd(total, file.SizeBytes, "declared repository size")
		if err != nil {
			return Manifest{}, err
		}
		if total > limits.MaxTotalBytes {
			return Manifest{}, fmt.Errorf(
				"repository is %d bytes, total limit is %d", total, limits.MaxTotalBytes,
			)
		}
		for _, object := range file.Objects() {
			if previous, found := objects[object.Digest]; found && previous != object.SizeBytes {
				return Manifest{}, fmt.Errorf(
					"object %s declared with two sizes (%d and %d)",
					object.Digest, previous, object.SizeBytes,
				)
			}
			objects[object.Digest] = object.SizeBytes
		}
	}
	if len(objects) > limits.MaxObjects {
		return Manifest{}, fmt.Errorf(
			"manifest has %d distinct objects, limit is %d", len(objects), limits.MaxObjects,
		)
	}
	return manifest, nil
}

// Planner validates a full declaration and grants only missing objects.
type Planner struct {
	Store       Store
	Claims      ClaimGate
	Limits      Limits
	GrantTTL    time.Duration
	GrantExpiry func(needBytes int64) (time.Time, error)
	Now         func() time.Time
}

func requireCompleteResidency(objects []Object, answer map[Ref]bool, label string) error {
	if len(answer) != len(objects) {
		return fmt.Errorf("examined %d of %d %s objects; residency is unknown", len(answer), len(objects), label)
	}
	for _, object := range objects {
		if _, found := answer[object.Digest]; !found {
			return fmt.Errorf("%s object %s has unknown residency", label, object.Digest)
		}
	}
	return nil
}

func checkedAdd(total, increment int64, label string) (int64, error) {
	if increment > math.MaxInt64-total {
		return 0, fmt.Errorf("%s exceeds the v1 signed 64-bit byte limit", label)
	}
	return total + increment, nil
}

// StagedKey returns the portable algorithm-qualified, session-scoped key for a
// pending object.
func StagedKey(sessionID string, ref Ref) (string, error) {
	session := strings.ToLower(strings.TrimSpace(sessionID))
	if !sessionPattern.MatchString(session) {
		return "", errors.New("session id must be one [a-z0-9][a-z0-9._-]{0,63} segment")
	}
	if ref.hex == "" {
		return "", errors.New("staged key requires a non-zero ref")
	}
	return path.Join("staging", "sha256", session, ref.hex), nil
}

// Plan validates a manifest, checks every distinct object, and returns upload
// grants for objects that are neither claimable-resident nor already staged.
func (p Planner) Plan(ctx context.Context, sessionID string, manifest Manifest) (Plan, error) {
	if p.Store == nil {
		return Plan{}, errors.New("tensorfs: store is required")
	}
	manifest, err := ValidateManifest(manifest, p.Limits)
	if err != nil {
		return Plan{}, err
	}
	sessionID = strings.ToLower(strings.TrimSpace(sessionID))
	if !sessionPattern.MatchString(sessionID) {
		return Plan{}, errors.New("session id must be one [a-z0-9][a-z0-9._-]{0,63} segment")
	}

	result := Plan{SessionID: sessionID, Manifest: manifest, DeclaredFiles: len(manifest.Files)}
	order := make([]Ref, 0)
	sizes := map[Ref]int64{}
	for _, file := range manifest.Files {
		result.DeclaredBytes, err = checkedAdd(result.DeclaredBytes, file.SizeBytes, "declared repository size")
		if err != nil {
			return Plan{}, err
		}
		for _, object := range file.Objects() {
			if previous, found := sizes[object.Digest]; found {
				if previous != object.SizeBytes {
					return Plan{}, fmt.Errorf(
						"object %s declared with two sizes (%d and %d)",
						object.Digest, previous, object.SizeBytes,
					)
				}
				continue
			}
			sizes[object.Digest] = object.SizeBytes
			order = append(order, object.Digest)
		}
	}
	result.DistinctObjects = len(order)
	objects := make([]Object, 0, len(order))
	for _, ref := range order {
		objects = append(objects, Object{Digest: ref, SizeBytes: sizes[ref]})
	}

	resident, err := p.Store.Residency(ctx, objects)
	if err != nil {
		return Plan{}, fmt.Errorf("residency: %w", err)
	}
	if err := requireCompleteResidency(objects, resident, "declared"); err != nil {
		return Plan{}, err
	}
	result.ExaminedObjects = len(objects)

	unclaimable := map[Ref]bool{}
	if p.Claims != nil {
		claimableCandidates := make([]Ref, 0)
		for _, ref := range order {
			if resident[ref] {
				claimableCandidates = append(claimableCandidates, ref)
			}
		}
		denied, claimErr := p.Claims.Unclaimable(ctx, claimableCandidates)
		if claimErr != nil {
			return Plan{}, fmt.Errorf("claim gate: %w", claimErr)
		}
		for _, ref := range denied {
			if !resident[ref] {
				return Plan{}, fmt.Errorf("claim gate returned undeclared or non-resident object %s", ref)
			}
			unclaimable[ref] = true
		}
	}

	pending := make([]Object, 0)
	for _, ref := range order {
		if resident[ref] && !unclaimable[ref] {
			result.Have = append(result.Have, ref)
			result.ResidentObjects++
			continue
		}
		pending = append(pending, Object{Digest: ref, SizeBytes: sizes[ref]})
	}
	staged, err := p.Store.StagedResidency(ctx, sessionID, pending)
	if err != nil {
		return Plan{}, fmt.Errorf("staged residency: %w", err)
	}
	if err := requireCompleteResidency(pending, staged, "staged"); err != nil {
		return Plan{}, err
	}

	var needBytes int64
	for _, object := range pending {
		if !staged[object.Digest] {
			needBytes, err = checkedAdd(needBytes, object.SizeBytes, "missing object size")
			if err != nil {
				return Plan{}, err
			}
		}
	}
	now := time.Now().UTC()
	if p.Now != nil {
		now = p.Now().UTC()
	}
	ttl := p.GrantTTL
	if ttl <= 0 {
		ttl = defaultGrantTTL
	}
	if p.GrantExpiry != nil && needBytes > 0 {
		deadline, expiryErr := p.GrantExpiry(needBytes)
		if expiryErr != nil {
			return Plan{}, expiryErr
		}
		ttl = deadline.Sub(now)
		if ttl <= 0 {
			return Plan{}, errors.New("grant deadline is already past")
		}
	}

	for _, object := range pending {
		key, keyErr := StagedKey(sessionID, object.Digest)
		if keyErr != nil {
			return Plan{}, keyErr
		}
		stagedObject := StagedObject{Object: object, StagingKey: key}
		if staged[object.Digest] {
			result.Staged = append(result.Staged, stagedObject)
			result.StagedObjects++
			continue
		}
		url, headers, grantErr := p.Store.PresignPut(ctx, stagedObject, ttl)
		if grantErr != nil {
			return Plan{}, fmt.Errorf("grant for %s: %w", object.Digest, grantErr)
		}
		if strings.TrimSpace(url) == "" {
			return Plan{}, fmt.Errorf("grant for %s has an empty URL", object.Digest)
		}
		result.Need = append(result.Need, Grant{
			StagedObject: stagedObject,
			URL:          url,
			Headers:      headers,
			ExpiresAt:    now.Add(ttl),
		})
	}
	sort.Slice(result.Have, func(i, j int) bool { return result.Have[i].String() < result.Have[j].String() })
	return result, nil
}
