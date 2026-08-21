package tensorfs

// QUANT RULES: a function on topologies, written once per FORMAT, ever.
//
// A rule is three things and nothing else:
//
//  1. an ELIGIBILITY PREDICATE computable from the topology alone — key
//     spelling, rank, dims, and the reference element type;
//  2. per-tensor EMISSIONS whose shapes are pure functions of the source dims
//     (`d0`, `d1/2`, `d1/16`), with optional-and-fixed-shape for the tensors a
//     calibration may or may not produce;
//  3. the DEQUANT equation, which is what makes derivability computable rather
//     than a table someone maintains.
//
// CONVENTION FACTS ARE IDENTITY, NOT METADATA. `cozy.nvfp4-flat` and
// `bfl.nvfp4-preswizzled` have the same element type, the same tensor names and
// the same ranks; they differ in nibble order and scale layout. Reading one as
// the other measured LPIPS 1.11 — every name, dtype and shape correct and every
// pixel wrong. So the conventions are in the digest, and the two rules can
// never alias.
//
// LAYOUT = rule(topology) is COMPUTED here and stored nowhere. This is the ONE
// evaluator; the hub calls it and never grows a second one.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// LayoutTensor is one entry of a computed expected header.
type LayoutTensor struct {
	// Dtypes is the ACCEPTED set, usually one. It is a set because a
	// reference-tolerant plain rule accepts an F32 island at either the
	// island's own dtype or the lane's — both are shipped, and refusing the
	// one the reference itself carries would refuse the reference.
	Dtypes []string
	Shape  Shape
	// Optional marks a tensor a calibration may or may not have produced
	// (`input_scale`, `pre_quant_scale`). Its SHAPE is still fixed: optional
	// means "absent or exactly this", never "anything".
	Optional bool
}

func (t LayoutTensor) String() string {
	suffix := ""
	if t.Optional {
		suffix = " (optional)"
	}
	return fmt.Sprintf("%s %s%s", t.Shape, strings.Join(t.Dtypes, "|"), suffix)
}

func (t LayoutTensor) accepts(dtype string) bool {
	for _, allowed := range t.Dtypes {
		if allowed == dtype {
			return true
		}
	}
	return false
}

// LayoutComponent is one component's computed header.
type LayoutComponent struct {
	Name    string
	Role    ComponentRole
	Tensors map[string]LayoutTensor
	keys    []string
}

// Keys in canonical order.
func (c *LayoutComponent) Keys() []string { return c.keys }

// Layout is quant(topology): the expected header of a checkpoint carrying this
// stamp. ALWAYS computed, never stored.
type Layout struct {
	ID         LayoutID
	Components []LayoutComponent
	byName     map[string]*LayoutComponent
	// transformed counts the tensors the rule actually acted on. ZERO IS A
	// REFUSAL TO STAMP, not a curiosity: a transforming rule that finds nothing
	// eligible computes a layout identical to its base plain rule's, and two
	// rules with one layout would make the stamp of a bf16 checkpoint depend on
	// catalog iteration order. An nvfp4 stamp on a tree with no nvfp4 tensor in
	// it is exactly the silent mis-identification v2 exists to make impossible.
	transformed int
}

// Transformed counts the tensors this layout's rule acted on.
func (l *Layout) Transformed() int { return l.transformed }

// Component looks one up by name.
func (l *Layout) Component(name string) *LayoutComponent { return l.byName[name] }

// Tensors counts every entry across every component.
func (l *Layout) Tensors() int {
	total := 0
	for at := range l.Components {
		total += len(l.Components[at].Tensors)
	}
	return total
}

// --- the document ----------------------------------------------------------

// Eligibility is the predicate half, transcribed from the PRODUCER rather than
// described. Every field is computable from a topology entry.
type Eligibility struct {
	// SourceDtypes bounds what the producer will convert; a rule that quantized
	// an I64 position-id table would be transcribing itself wrong.
	SourceDtypes []string `json:"source_dtypes,omitempty"`
	// Rank pins the rank exactly; RankAtLeast bounds it below. The scale-free
	// fp8 storage flavour converts every float `.weight` of rank 2 OR MORE
	// (conv kernels included), where the w8a8 requant is rank-2 only — that is
	// a real difference between two producers and it has to be expressible.
	Rank        int    `json:"rank,omitempty"`
	RankAtLeast int    `json:"rank_at_least,omitempty"`
	KeySuffix   string `json:"key_suffix,omitempty"`
	// DimAlign is per-axis, outermost first. `[16, 32]` is nvfp4's
	// out%16 / in%32 pair; 0 means "no constraint on this axis".
	DimAlign []uint64 `json:"dim_align,omitempty"`
	// RequireRepeatedBlockSegment is the `\.\d+\.` conjunct: only weights under
	// a repeated block convert, so embeddings, final norms and heads keep
	// source precision.
	RequireRepeatedBlockSegment bool `json:"require_repeated_block_segment,omitempty"`
	// SkipModuleSubstrings and SkipModuleExact are the producer's skip list,
	// split by how it anchors: `embed` matches anywhere in the module path,
	// `^proj_in$` matches only a module path that IS `proj_in` — which is why
	// SDXL's `down_blocks.0.attentions.0.proj_in` DOES convert.
	SkipModuleSubstrings []string `json:"skip_module_substrings,omitempty"`
	SkipModuleExact      []string `json:"skip_module_exact,omitempty"`
}

// Emission is one tensor a rule produces per eligible source tensor.
type Emission struct {
	// Key is a template over `{module}` (the source key minus the eligibility
	// suffix) and `{key}` (the whole source key).
	Key      string   `json:"key"`
	Dtype    string   `json:"dtype"`
	Shape    []string `json:"shape"`
	Optional bool     `json:"optional,omitempty"`

	formula []*shapeExpr
}

// QuantRule is one validated rule document.
type QuantRule struct {
	handle      Handle
	description string
	// DeclaredDtype is the torch-precise spelling of the LANE's quantization —
	// the field gen-worker's `ctx.lane.dtype` reads. Under v1 this lived on a
	// contract document and the sm floor was looked up from a table keyed on
	// the string; under v2 both live HERE, on the rule identity, so a lane
	// cannot lose its floor by being spelled differently.
	DeclaredDtype string
	// CapabilityFloorSM is the minimum CUDA capability the format needs,
	// x10 (89 = sm89, 100 = sm100/Blackwell). 0 means "any card".
	CapabilityFloorSM int
	// BaseDtype is what NON-eligible tensors carry. `@reference` means "keep
	// whatever the reference packaging had", which is what a plain rule over an
	// already-mixed tree means.
	BaseDtype string
	// ReferenceTolerant lets an island keep the reference's own dtype.
	ReferenceTolerant bool
	// Conventions are IDENTITY. Nibble order, scale layout, scale span: the
	// facts two rules can agree on every name and shape and still differ by.
	Conventions map[string]string
	// ScopeRoles are the component roles the rule transforms; every other
	// component passes through. Empty means "the whole checkpoint".
	ScopeRoles []ComponentRole
	// Lossy decides the storage tier: lossy means gate, produce once, keep
	// provenance. A cast is usually lossy too (fp16->bf16 drops mantissa) and
	// is waved through the gate cheaply, not exempted from it.
	Lossy bool
	// Inverse is the dequant equation — the derivability arrow, in one line.
	Inverse string

	eligible  *Eligibility
	emissions []Emission
}

// Handle, Description, Digest.
func (r *QuantRule) Handle() Handle      { return r.handle }
func (r *QuantRule) Description() string { return r.description }
func (r *QuantRule) Digest() string      { return digestOf(r.canonical()) }

// Transforms reports whether the rule changes anything at all. A plain rule
// does not, which is what makes plain-to-plain derivation a pure cast.
func (r *QuantRule) Transforms() bool { return r.eligible != nil && len(r.emissions) > 0 }

func (r *QuantRule) canonical() string {
	var out strings.Builder
	out.WriteString(QuantRuleFormat)
	fmt.Fprintf(&out, "\nname=%s\nversion=%d\n", r.handle.Name, r.handle.Version)
	fmt.Fprintf(&out, "declared_dtype=%s\nfloor_sm=%d\nbase=%s\ntolerant=%t\nlossy=%t\n",
		r.DeclaredDtype, r.CapabilityFloorSM, r.BaseDtype, r.ReferenceTolerant, r.Lossy)
	for _, key := range sortedMapKeys(r.Conventions) {
		fmt.Fprintf(&out, "convention %s=%s\n", key, r.Conventions[key])
	}
	for _, role := range r.ScopeRoles {
		fmt.Fprintf(&out, "scope=%s\n", role)
	}
	if r.eligible != nil {
		fmt.Fprintf(&out, "eligible dtypes=%s rank=%d rank>=%d suffix=%s align=%v block=%t\n",
			strings.Join(r.eligible.SourceDtypes, ","), r.eligible.Rank,
			r.eligible.RankAtLeast, r.eligible.KeySuffix, r.eligible.DimAlign,
			r.eligible.RequireRepeatedBlockSegment)
		fmt.Fprintf(&out, "skip substrings=%s exact=%s\n",
			strings.Join(r.eligible.SkipModuleSubstrings, ","),
			strings.Join(r.eligible.SkipModuleExact, ","))
	}
	for _, emission := range r.emissions {
		fmt.Fprintf(&out, "emit key=%s dtype=%s shape=%s optional=%t\n",
			emission.Key, emission.Dtype, strings.Join(emission.Shape, ","), emission.Optional)
	}
	fmt.Fprintf(&out, "inverse=%s\n", r.Inverse)
	return out.String()
}

type rawQuantRule struct {
	Format            *string           `json:"format"`
	Name              *string           `json:"name"`
	Version           *uint32           `json:"version"`
	Description       string            `json:"description"`
	DeclaredDtype     string            `json:"declared_dtype"`
	CapabilityFloorSM int               `json:"capability_floor_sm"`
	BaseDtype         string            `json:"base_dtype"`
	ReferenceTolerant bool              `json:"reference_tolerant,omitempty"`
	Conventions       map[string]string `json:"conventions"`
	ScopeRoles        []string          `json:"scope_roles,omitempty"`
	Lossy             bool              `json:"lossy"`
	Inverse           string            `json:"inverse,omitempty"`
	Eligible          *Eligibility      `json:"eligible,omitempty"`
	Emissions         []Emission        `json:"emissions,omitempty"`
	Digest            string            `json:"digest,omitempty"`
}

// ParseQuantRule reads and validates one rule document.
func ParseQuantRule(document []byte) (*QuantRule, error) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.DisallowUnknownFields()
	var raw rawQuantRule
	if err := decoder.Decode(&raw); err != nil {
		return nil, refuse("json", "%v", err)
	}
	if raw.Format == nil || *raw.Format != QuantRuleFormat {
		return nil, refuse("format", "not a %s document", QuantRuleFormat)
	}
	if raw.Name == nil || raw.Version == nil {
		return nil, refuse("identity", "a quant rule is named and versioned")
	}
	handle, err := ParseHandle(fmt.Sprintf("%s@%d", *raw.Name, *raw.Version))
	if err != nil {
		return nil, err
	}
	if raw.DeclaredDtype == "" {
		return nil, refuse("declared-dtype",
			"%s declares no dtype — a serve lane that cannot say what its "+
				"quantization IS is undeclarable, and gen-worker derives its "+
				"capability floor from this field alone", handle)
	}
	if raw.BaseDtype == "" {
		return nil, refuse("base-dtype", "%s declares no base dtype", handle)
	}
	rule := &QuantRule{
		handle: handle, description: raw.Description,
		DeclaredDtype: raw.DeclaredDtype, CapabilityFloorSM: raw.CapabilityFloorSM,
		BaseDtype: raw.BaseDtype, ReferenceTolerant: raw.ReferenceTolerant,
		Conventions: raw.Conventions, Lossy: raw.Lossy, Inverse: raw.Inverse,
		eligible: raw.Eligible,
	}
	for _, role := range raw.ScopeRoles {
		rule.ScopeRoles = append(rule.ScopeRoles, ComponentRole(role))
	}
	if (raw.Eligible == nil) != (len(raw.Emissions) == 0) {
		return nil, refuse("rule",
			"%s has an eligibility predicate without emissions or the reverse — "+
				"half a rule computes half a header", handle)
	}
	if raw.Eligible != nil && len(rule.Conventions) == 0 {
		return nil, refuse("conventions",
			"%s transforms tensors but states no convention facts. Nibble order, "+
				"scale layout and scale span are rule IDENTITY: two formats that "+
				"agree on every name and shape and differ here measured LPIPS 1.11 "+
				"when they were conflated", handle)
	}
	for at := range raw.Emissions {
		emission := raw.Emissions[at]
		if emission.Key == "" || emission.Dtype == "" || len(emission.Shape) == 0 {
			return nil, refuse("emission", "%s has an incomplete emission", handle)
		}
		if !strings.Contains(emission.Key, "{module}") && !strings.Contains(emission.Key, "{key}") {
			return nil, refuse("emission",
				"%s emits %q, which names no source tensor", handle, emission.Key)
		}
		for _, term := range emission.Shape {
			expression, err := parseShapeExpr(term)
			if err != nil {
				return nil, err
			}
			emission.formula = append(emission.formula, expression)
		}
		rule.emissions = append(rule.emissions, emission)
	}
	if raw.Digest != "" && raw.Digest != rule.Digest() {
		return nil, refuse("digest", "%s carries digest %s but hashes to %s",
			handle, raw.Digest, rule.Digest())
	}
	return rule, nil
}

// --- the predicate ---------------------------------------------------------

func (r *QuantRule) scopes(role ComponentRole) bool {
	if len(r.ScopeRoles) == 0 {
		return true
	}
	for _, scoped := range r.ScopeRoles {
		if scoped == role {
			return true
		}
	}
	return false
}

// referenceDtype is what the reference checkpoint carried for one key.
func referenceDtype(topology *Topology, component *TopologyComponent, key string) string {
	if island, found := component.Islands[key]; found {
		return island
	}
	return topology.dtype
}

// Eligible reports whether one topology entry is transformed by this rule.
// Computable from the topology alone — that property is what lets a layout be
// computed without ever reading a weight byte.
func (r *QuantRule) Eligible(topology *Topology, component *TopologyComponent, key string, shape Shape) bool {
	if r.eligible == nil || !r.scopes(component.Role) {
		return false
	}
	predicate := r.eligible
	if len(predicate.SourceDtypes) > 0 {
		found := false
		for _, dtype := range predicate.SourceDtypes {
			if dtype == referenceDtype(topology, component, key) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if predicate.Rank != 0 && len(shape) != predicate.Rank {
		return false
	}
	if predicate.RankAtLeast != 0 && len(shape) < predicate.RankAtLeast {
		return false
	}
	if predicate.KeySuffix != "" && !strings.HasSuffix(key, predicate.KeySuffix) {
		return false
	}
	for axis, alignment := range predicate.DimAlign {
		if alignment == 0 {
			continue
		}
		if axis >= len(shape) || shape[axis]%alignment != 0 {
			return false
		}
	}
	if predicate.RequireRepeatedBlockSegment && !hasRepeatedBlockSegment(key) {
		return false
	}
	module := strings.TrimSuffix(key, predicate.KeySuffix)
	for _, needle := range predicate.SkipModuleSubstrings {
		if strings.Contains(module, needle) {
			return false
		}
	}
	for _, exact := range predicate.SkipModuleExact {
		if module == exact {
			return false
		}
	}
	return true
}

// hasRepeatedBlockSegment is the producer's `\.\d+\.`: a key lives under a
// repeated block iff some dot-separated segment is all digits and is neither
// first nor last.
func hasRepeatedBlockSegment(key string) bool {
	segments := strings.Split(key, ".")
	for at := 1; at < len(segments)-1; at++ {
		if segments[at] == "" {
			continue
		}
		digits := true
		for _, character := range segments[at] {
			if character < '0' || character > '9' {
				digits = false
				break
			}
		}
		if digits {
			return true
		}
	}
	return false
}

// --- the computation -------------------------------------------------------

// Apply computes quant(topology): the expected header, entry by entry.
//
// This is the whole of v2's compression. There is no stored document for
// `sdxl.diffusers + cozy.fp8-rowwise`; there is a topology, a rule, and this
// function.
func (r *QuantRule) Apply(topology *Topology) (*Layout, error) {
	layout := &Layout{
		ID:     LayoutID{Topology: topology.handle, Quant: r.handle, Bytes: DefaultLayout},
		byName: map[string]*LayoutComponent{},
	}
	for at := range topology.components {
		component := &topology.components[at]
		computed := LayoutComponent{
			Name: component.Name, Role: component.Role,
			Tensors: make(map[string]LayoutTensor, len(component.tensors)),
		}
		for _, key := range component.keys {
			shape := component.tensors[key]
			if !r.Eligible(topology, component, key, shape) {
				computed.Tensors[key] = LayoutTensor{
					Dtypes: r.passthroughDtypes(topology, component, key), Shape: shape.clone(),
				}
				continue
			}
			module := strings.TrimSuffix(key, r.eligible.KeySuffix)
			for index := range r.emissions {
				emission := &r.emissions[index]
				emitted := strings.NewReplacer("{module}", module, "{key}", key).
					Replace(emission.Key)
				emittedShape, err := evalShape(emission.formula, shape)
				if err != nil {
					return nil, refuse("emission",
						"%s applying %s to %q %s: %v",
						topology.handle, r.handle, key, shape, err)
				}
				if _, collision := computed.Tensors[emitted]; collision {
					return nil, refuse("emission",
						"%s applying %s: %q is emitted twice", topology.handle, r.handle, emitted)
				}
				computed.Tensors[emitted] = LayoutTensor{
					Dtypes: []string{emission.Dtype}, Shape: emittedShape,
					Optional: emission.Optional,
				}
			}
			layout.transformed++
		}
		computed.keys = sortedMapKeys(computed.Tensors)
		layout.Components = append(layout.Components, computed)
	}
	for at := range layout.Components {
		layout.byName[layout.Components[at].Name] = &layout.Components[at]
	}
	return layout, nil
}

// passthroughDtypes is what an untransformed tensor may carry.
//
// The reference-tolerance is the F32-island rule: a bf16 tree that ships one
// F32 `logit_scale` is still the bf16 packaging, and a lane that refused it
// would refuse the reference checkpoint itself.
func (r *QuantRule) passthroughDtypes(topology *Topology, component *TopologyComponent, key string) []string {
	reference := referenceDtype(topology, component, key)
	if r.BaseDtype == "@reference" {
		return []string{reference}
	}
	if r.ReferenceTolerant && reference != r.BaseDtype && !isQuantElement(reference) {
		if _, island := component.Islands[key]; island {
			return []string{r.BaseDtype, reference}
		}
	}
	return []string{r.BaseDtype}
}

// isQuantElement separates a COMPUTE element type from a QUANTIZED one.
//
// Reference tolerance exists for the first kind and must never reach the
// second. Real trees mix compute types freely — LTX-2.3 ships a bf16 denoiser
// beside an fp32 text encoder, Wan 2.2 the reverse — and a plain lane serves
// either. An fp8 or packed-nibble element at a key is not a wider or narrower
// spelling of the same number; it is a QUANTIZATION, and tolerating it would
// let a genuinely quantized checkpoint stamp as `plain.bf16` and be served as
// if its scales did not exist.
func isQuantElement(dtype string) bool {
	switch dtype {
	case "F64", "F32", "F16", "BF16":
		return false
	case "I64", "I32", "I16", "BOOL":
		// Index tables and masks: not compute precision, and not quantization
		// either. A plain lane carries them exactly as the reference did.
		return false
	default:
		return true
	}
}
