package tensorfs

// MORPHISMS: topology -> topology maps. With quant rules, the ONLY
// hand-authored, human-ratified artifacts in v2.
//
// Two kinds, and the difference is what it COSTS to run them:
//
//	REKEY   a renaming. Same tensors, same bytes, new names. T1: a new
//	        manifest over the same chunks, ZERO new bytes in CAS.
//	SEAM    a fusion or a split. `q_proj`|`k_proj`|`v_proj` <-> `qkv_proj`.
//	        Contiguous seams are T2 (manifest range-slicing, same chunks);
//	        INTERLEAVED ones — head-major groups — are T3 and must be
//	        materialized once.
//
// A SEAM FILE CARRIES ITS EVIDENCE. The fusion facts here are not derivable
// from any header: a [2048, 2048] tensor cannot say whether it is q|k|v at
// 1:1:2 or in equal thirds, and getting it wrong is silent — every name, dtype
// and shape correct and every number wrong. So each seam carries the ratified
// provenance block that proved it, transplanted verbatim from the v1 document
// it was ratified in (tensorfs#150). The facts SURVIVE the cut; only the
// container changed.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// StorageTier prices a derivation. Paul-ratified: lossless derives freely,
// lossy gates.
type StorageTier int

const (
	// TierRekey is a new manifest over the same chunks — zero new bytes.
	TierRekey StorageTier = 1
	// TierContiguous is manifest range-slicing — still the same chunks.
	TierContiguous StorageTier = 2
	// TierInterleaved needs load-time materialization; store ONE packaging.
	TierInterleaved StorageTier = 3
	// TierQuant is lossy: gate, produce once, keep provenance.
	TierQuant StorageTier = 4
)

func (t StorageTier) String() string {
	switch t {
	case TierRekey:
		return "T1 rekey (zero new bytes)"
	case TierContiguous:
		return "T2 contiguous seam (manifest range-slicing)"
	case TierInterleaved:
		return "T3 interleaved seam (materialized once)"
	case TierQuant:
		return "T4 quant (lossy, gated, produced once)"
	default:
		return fmt.Sprintf("T%d", int(t))
	}
}

// Rekey is one row of a rekey table: either a PREFIX rename (`net.` ->
// `model.diffusion_model.`, which covers a whole packaging in one row) or a
// KEY PATTERN rename with `{i}` holes, for the leaf spellings two packagings
// disagree about (`...attn.to_out.0.weight` -> `...attn.out_proj.weight`).
//
// Rows are tried in order and the first that matches wins, so a table can put
// its specific leaf renames above its blanket prefix rename.
type Rekey struct {
	FromPrefix string `json:"from_prefix,omitempty"`
	ToPrefix   string `json:"to_prefix,omitempty"`
	FromKey    string `json:"from_key,omitempty"`
	ToKey      string `json:"to_key,omitempty"`
}

func (r Rekey) apply(key string) (string, bool) {
	if r.FromKey != "" {
		holes, matched := matchHoles(r.FromKey, key)
		if !matched {
			return "", false
		}
		return renderHoles(r.ToKey, holes), true
	}
	if r.FromPrefix != "" && strings.HasPrefix(key, r.FromPrefix) {
		return r.ToPrefix + key[len(r.FromPrefix):], true
	}
	return "", false
}

func (r Rekey) inverse() Rekey {
	return Rekey{
		FromPrefix: r.ToPrefix, ToPrefix: r.FromPrefix,
		FromKey: r.ToKey, ToKey: r.FromKey,
	}
}

// SeamPart is one share of a fused axis.
type SeamPart struct {
	Role  string `json:"role"`
	Share uint64 `json:"share"`
	// Key is the SPLIT-side key of this part, `{i}` holes included.
	Key string `json:"key"`
}

// Seam is one fusion fact.
type Seam struct {
	// Target is the FUSED key (the `to` side of the morphism).
	Target string `json:"target"`
	// Parts are the split-side keys, in the order they concatenate.
	Parts []SeamPart `json:"parts"`
	// Groups > 1 means HEAD-MAJOR interleaving: the axis is cut into Groups
	// repetitions of the part pattern, not into one run per part. This is the
	// fact a flat split gets right for group 0 and wrong for the other 15.
	Groups uint64 `json:"groups"`
	// Component scopes the seam; empty means every component.
	Component string `json:"component,omitempty"`
	// Provenance is the RATIFIED evidence: what proved these shares, cited to
	// the module source line. Required — a seam without it is a guess.
	Provenance string `json:"provenance"`
}

// Interleaved reports the T3 case.
func (s *Seam) Interleaved() bool { return s.Groups > 1 }

// Morphism is one validated topology -> topology map.
type Morphism struct {
	handle      Handle
	description string
	from, to    Handle
	tier        StorageTier
	invertible  bool
	rekeys      []Rekey
	seams       []Seam
	provenance  []string
	// inverted flips every seam from a fusion to a split. Set only by Inverse.
	inverted bool
}

func (m *Morphism) Handle() Handle       { return m.handle }
func (m *Morphism) From() Handle         { return m.from }
func (m *Morphism) To() Handle           { return m.to }
func (m *Morphism) Tier() StorageTier    { return m.tier }
func (m *Morphism) Invertible() bool     { return m.invertible }
func (m *Morphism) Seams() []Seam        { return m.seams }
func (m *Morphism) Provenance() []string { return m.provenance }
func (m *Morphism) Digest() string       { return digestOf(m.canonical()) }

func (m *Morphism) canonical() string {
	var out strings.Builder
	out.WriteString(MorphismFormat)
	fmt.Fprintf(&out, "\nname=%s\nversion=%d\nfrom=%s\nto=%s\ntier=%d\ninvertible=%t\n",
		m.handle.Name, m.handle.Version, m.from, m.to, int(m.tier), m.invertible)
	for _, rekey := range m.rekeys {
		fmt.Fprintf(&out, "rekey prefix %s->%s key %s->%s\n",
			rekey.FromPrefix, rekey.ToPrefix, rekey.FromKey, rekey.ToKey)
	}
	for _, seam := range m.seams {
		fmt.Fprintf(&out, "seam target=%s component=%s groups=%d\n",
			seam.Target, seam.Component, seam.Groups)
		for _, part := range seam.Parts {
			fmt.Fprintf(&out, "part role=%s share=%d key=%s\n", part.Role, part.Share, part.Key)
		}
	}
	return out.String()
}

type rawMorphism struct {
	Format      *string  `json:"format"`
	Name        *string  `json:"name"`
	Version     *uint32  `json:"version"`
	Description string   `json:"description"`
	From        string   `json:"from"`
	To          string   `json:"to"`
	Tier        int      `json:"tier"`
	Invertible  bool     `json:"invertible"`
	Rekey       []Rekey  `json:"rekey,omitempty"`
	Seams       []Seam   `json:"seams,omitempty"`
	Provenance  []string `json:"provenance"`
	Digest      string   `json:"digest,omitempty"`
}

// ParseMorphism reads and validates one seam file.
func ParseMorphism(document []byte) (*Morphism, error) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.DisallowUnknownFields()
	var raw rawMorphism
	if err := decoder.Decode(&raw); err != nil {
		return nil, refuse("json", "%v", err)
	}
	if raw.Format == nil || *raw.Format != MorphismFormat {
		return nil, refuse("format", "not a %s document", MorphismFormat)
	}
	if raw.Name == nil || raw.Version == nil {
		return nil, refuse("identity", "a morphism is named and versioned")
	}
	handle, err := ParseHandle(fmt.Sprintf("%s@%d", *raw.Name, *raw.Version))
	if err != nil {
		return nil, err
	}
	from, err := ParseHandle(raw.From)
	if err != nil {
		return nil, err
	}
	to, err := ParseHandle(raw.To)
	if err != nil {
		return nil, err
	}
	if from == to {
		return nil, refuse("morphism", "%s maps %s to itself", handle, from)
	}
	if len(raw.Rekey) == 0 && len(raw.Seams) == 0 {
		return nil, refuse("morphism", "%s changes nothing", handle)
	}
	if len(raw.Provenance) == 0 {
		return nil, refuse("provenance",
			"%s carries no ratified evidence. A morphism is hand-authored; the "+
				"evidence that proved it is the only reason to trust it", handle)
	}
	morphism := &Morphism{
		handle: handle, description: raw.Description, from: from, to: to,
		tier: StorageTier(raw.Tier), invertible: raw.Invertible,
		rekeys: raw.Rekey, seams: raw.Seams, provenance: raw.Provenance,
	}
	for _, seam := range raw.Seams {
		if len(seam.Parts) < 2 {
			return nil, refuse("seam", "%s: %q fuses fewer than two parts", handle, seam.Target)
		}
		if seam.Provenance == "" {
			return nil, refuse("provenance",
				"%s: seam %q states no evidence. A [2048, 2048] tensor cannot say "+
					"whether it is q|k|v at 1:1:2 or in equal thirds, and getting it "+
					"wrong is silent", handle, seam.Target)
		}
		roles := map[string]bool{}
		for _, part := range seam.Parts {
			if part.Share == 0 || part.Key == "" || part.Role == "" {
				return nil, refuse("seam", "%s: %q has an incomplete part", handle, seam.Target)
			}
			if roles[part.Role] {
				return nil, refuse("seam", "%s: %q repeats role %q", handle, seam.Target, part.Role)
			}
			roles[part.Role] = true
		}
		if seam.Groups == 0 {
			return nil, refuse("seam", "%s: %q declares zero groups", handle, seam.Target)
		}
	}
	if morphism.tier < TierRekey || morphism.tier > TierInterleaved {
		return nil, refuse("tier", "%s declares tier %d", handle, raw.Tier)
	}
	if len(raw.Seams) == 0 && morphism.tier != TierRekey {
		return nil, refuse("tier",
			"%s only renames keys, which is T1 — a rename that claimed a higher "+
				"tier would price zero bytes as work", handle)
	}
	if raw.Digest != "" && raw.Digest != morphism.Digest() {
		return nil, refuse("digest", "%s carries digest %s but hashes to %s",
			handle, raw.Digest, morphism.Digest())
	}
	return morphism, nil
}

// Apply maps a topology through this morphism, producing the topology the `to`
// handle names. The result is CHECKED against the catalog's own record of that
// handle by the catalog loader, which is how a seam file that is subtly wrong
// becomes a red at load rather than a wrong admit at serve.
//
// A seam's parts are always spelled in the SPLIT packaging and its target in
// the FUSED one, whichever direction the morphism runs; the rekey table maps
// everything the seams do not touch. That way one seam file answers both
// questions and cannot write the shares down twice.
func (m *Morphism) Apply(topology *Topology) (*Topology, error) {
	if topology.handle != m.from {
		return nil, refuse("morphism", "%s maps %s, not %s", m.handle, m.from, topology.handle)
	}
	out := &Topology{
		handle: m.to, description: fmt.Sprintf("computed by %s from %s", m.handle, m.from),
		derivedFrom: topology.derivedFrom, dtype: topology.dtype,
		byName: map[string]*TopologyComponent{},
	}
	for at := range topology.components {
		source := &topology.components[at]
		mapped := TopologyComponent{
			Name: source.Name, Role: source.Role,
			tensors: map[string]Shape{},
		}
		if len(source.Islands) > 0 {
			mapped.Islands = map[string]string{}
		}
		seamed, err := m.runSeams(source)
		if err != nil {
			return nil, err
		}
		for _, key := range source.keys {
			if seamed.consumed[key] {
				continue
			}
			mapped.tensors[m.rekey(key)] = source.tensors[key].clone()
			if island, found := source.Islands[key]; found {
				mapped.Islands[m.rekey(key)] = island
			}
		}
		for key, shape := range seamed.produced {
			if _, collision := mapped.tensors[key]; collision {
				return nil, refuse("seam",
					"%s: %q is produced by a seam and also survives the rekey table",
					m.handle, key)
			}
			mapped.tensors[key] = shape
		}
		mapped.keys = sortedMapKeys(mapped.tensors)
		out.components = append(out.components, mapped)
	}
	for at := range out.components {
		out.byName[out.components[at].Name] = &out.components[at]
	}
	return out, nil
}

func (m *Morphism) rekey(key string) string {
	for _, rule := range m.rekeys {
		if mapped, matched := rule.apply(key); matched {
			return mapped
		}
	}
	return key
}

// Inverse is the morphism run backwards: the rekey table swapped, and every
// seam read as a SPLIT instead of a fusion.
//
// It exists so a fusion fact is authored ONCE, in the direction it was proved
// in ("q|k|v concatenate into `in_proj_weight` at 1:1:2"), and still answers
// the other question — a fused checkpoint arriving at a lane that wants the
// split packaging. Two hand-authored documents for one fact would be two
// chances to write the shares down differently.
func (m *Morphism) Inverse() *Morphism {
	out := &Morphism{
		handle:      Handle{Name: m.handle.Name + "-inverse", Version: m.handle.Version},
		description: "the inverse of " + m.handle.String(),
		from:        m.to,
		to:          m.from,
		tier:        m.tier,
		invertible:  true,
		seams:       m.seams,
		provenance:  m.provenance,
		inverted:    !m.inverted,
	}
	for _, rule := range m.rekeys {
		out.rekeys = append(out.rekeys, rule.inverse())
	}
	return out
}

type seamResult struct {
	consumed map[string]bool
	produced map[string]Shape
}

// runSeams applies every seam over one component, in whichever direction this
// morphism runs.
//
// A seam whose parts are not ALL present is not silently skipped: the source
// topology is not the packaging this morphism claims to map, and pretending
// otherwise produces a target topology nothing will ever match.
func (m *Morphism) runSeams(component *TopologyComponent) (seamResult, error) {
	result := seamResult{consumed: map[string]bool{}, produced: map[string]Shape{}}
	for index := range m.seams {
		seam := &m.seams[index]
		if seam.Component != "" && seam.Component != component.Name {
			continue
		}
		var err error
		if m.inverted {
			err = m.split(component, seam, &result)
		} else {
			err = m.fuse(component, seam, &result)
		}
		if err != nil {
			return result, err
		}
	}
	return result, nil
}

func seamUnits(seam *Seam) uint64 {
	units := uint64(0)
	for _, part := range seam.Parts {
		units += part.Share
	}
	return units
}

// fuse concatenates the split parts into the fused target.
func (m *Morphism) fuse(component *TopologyComponent, seam *Seam, result *seamResult) error {
	instances := map[string][]string{} // rendered target -> part keys in part order
	for _, part := range seam.Parts {
		for _, key := range component.keys {
			holes, matched := matchHoles(part.Key, key)
			if !matched {
				continue
			}
			target := renderHoles(seam.Target, holes)
			instances[target] = append(instances[target], key)
		}
	}
	for _, target := range sortedMapKeys(instances) {
		keys := instances[target]
		if len(keys) != len(seam.Parts) {
			return refuse("seam",
				"%s: %q would fuse %d of %d parts — the source packaging is not the "+
					"one this seam maps", m.handle, target, len(keys), len(seam.Parts))
		}
		var fused Shape
		for at, key := range keys {
			shape := component.tensors[key]
			if at == 0 {
				fused = shape.clone()
				continue
			}
			if len(shape) != len(fused) {
				return refuse("seam", "%s: %q fuses ranks %d and %d",
					m.handle, target, len(fused), len(shape))
			}
			for axis := 1; axis < len(shape); axis++ {
				if shape[axis] != fused[axis] {
					return refuse("seam",
						"%s: %q fuses tensors that disagree on axis %d (%d vs %d)",
						m.handle, target, axis, fused[axis], shape[axis])
				}
			}
			fused[0] += shape[0]
		}
		units := seamUnits(seam)
		if fused[0]%(units*seam.Groups) != 0 {
			return refuse("seam",
				"%s: %q fuses to %d rows, which is not %d group(s) of %d shares",
				m.handle, target, fused[0], seam.Groups, units)
		}
		for _, key := range keys {
			result.consumed[key] = true
		}
		result.produced[target] = fused
	}
	return nil
}

// split cuts the fused target back into its parts, by SHARE.
//
// The shares are the whole content of a seam file. A [8192, 2048] tensor cannot
// say whether it is q|k|v in equal thirds or at 1:1:2, and a wrong split is
// silent — which is why every seam carries the source citation that proved it.
func (m *Morphism) split(component *TopologyComponent, seam *Seam, result *seamResult) error {
	units := seamUnits(seam)
	for _, key := range component.keys {
		holes, matched := matchHoles(seam.Target, key)
		if !matched {
			continue
		}
		shape := component.tensors[key]
		if len(shape) == 0 {
			return refuse("seam", "%s: %q is a scalar and cannot be split", m.handle, key)
		}
		if shape[0]%(units*seam.Groups) != 0 {
			return refuse("seam",
				"%s: %q has %d rows, which is not %d group(s) of %d shares",
				m.handle, key, shape[0], seam.Groups, units)
		}
		unit := shape[0] / units
		for _, part := range seam.Parts {
			piece := shape.clone()
			piece[0] = unit * part.Share
			result.produced[renderHoles(part.Key, holes)] = piece
		}
		result.consumed[key] = true
	}
	return nil
}

// matchHoles matches a `{i}` pattern against a concrete key, returning the hole
// values. Leading zeros refuse, so one tensor name has one spelling.
func matchHoles(pattern, name string) ([]string, bool) {
	var holes []string
	rest, text := name, pattern
	for {
		at := strings.Index(text, "{i}")
		if at < 0 {
			return holes, rest == text
		}
		literal := text[:at]
		if !strings.HasPrefix(rest, literal) {
			return nil, false
		}
		rest = rest[len(literal):]
		digits := 0
		for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
			digits++
		}
		if digits == 0 || (digits > 1 && rest[0] == '0') {
			return nil, false
		}
		holes = append(holes, rest[:digits])
		rest = rest[digits:]
		text = text[at+3:]
	}
}

func renderHoles(pattern string, holes []string) string {
	out := pattern
	for _, hole := range holes {
		out = strings.Replace(out, "{i}", hole, 1)
	}
	return out
}
