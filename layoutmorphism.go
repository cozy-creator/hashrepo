package tensorfs

// LAYOUT MORPHISMS: where a tensor's elements SIT IN STORAGE, as data.
//
// A topology says WHICH tensors exist and what shape they are; a quant rule
// says what element type they are stored in. Neither says anything about the
// ORDER the elements are written in — and that order is where quantized fast
// paths and custom kernels live. Today every layer repacks by hand: inductor
// re-derives NHWC conv weights every forward call, nunchaku's loader unswizzles
// seven sub-axes of a scale tensor in Python, and two byte-different packagings
// of one checkpoint are stored twice because nothing can say they are the same
// tensors in a different arrangement.
//
// A layout morphism says it, in one closed form:
//
//	FACTOR the logical shape into sub-axes, PERMUTE the sub-axes, write
//	row-major. A sub-axis product that EXCEEDS its logical dim is padding.
//
// That is the whole language, and every entry in the v1 vocabulary is one
// instance of it:
//
//	contiguous            the identity — no factorization, no permutation
//	channels_last (2d/3d) NCHW -> NHWC, NCDHW -> NDHWC
//	transposed            a rank-2 swap
//	stride-padding        the innermost dim rounded up to an alignment
//	cublas.blockscale     128x4-tiled nvfp4 block scales (padded AND blocked)
//	nunchaku.micro-scale  a seven-sub-axis warp-lane interleave
//
// WHY THIS FORM AND NOT CODE. A morphism written as code is ratified by
// review, and review is exactly what missed nunchaku's `(4, 4, 8)` row split
// reshaping without error into `(4, 8, 4)` — every name, dtype and shape
// correct and every channel silently permuted. Written as data, the map is
// MACHINE-VERIFIABLE: apply it and apply its inverse and compare bytes. A
// record that survives that is AUTO-RATIFIED; no human signs a permutation.
//
// THE TRANSFORM ITSELF IS NOT HERE. This file decides — parses, validates,
// prices and plans. The bytes are moved in exactly one place, the Rust crate's
// `layout_morphism` module, which has two backends (CPU materialization and
// the GPU fill) over ONE implementation. A Go copy of the applier would be a
// second answer to "which byte goes where", which is the disease v1's three
// matchers already proved is undetectable until a pod serves noise.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// LayoutMorphismFormat tags the record class.
const LayoutMorphismFormat = "tensorfs-layout-morphism-v2"

// DefaultLayout is the byte arrangement every tree carries unless it says
// otherwise: plain row-major. It is the identity morphism, and it is the
// coordinate every stamp written before this file existed means.
var DefaultLayout = Handle{Name: "torch.contiguous", Version: 1}

// LayoutClass sizes a morphism, and the two classes are worth different money.
type LayoutClass string

const (
	// ClassInductor is the compiler's own closed vocabulary. Worthwhile but
	// small: inductor races kernels, not layouts, and its layout pass is
	// rule-based.
	ClassInductor LayoutClass = "inductor"
	// ClassEndpointDeclared is the kernel-library class — the arrangement a
	// quantized fast path or a custom kernel demands. These are the real wins.
	ClassEndpointDeclared LayoutClass = "endpoint-declared"
)

// SubAxis is one factor of one logical axis.
//
// Extent is a formula in the ONE shape language v2 already has (`d0`, `d1/16`,
// literals, `ceil(...)`), so a padded factorization spells its rounding the
// same way a quant rule's emission does. That is not a convenience: the cuBLAS
// block-scale layout's element count and `bfl.nvfp4-preswizzled@1`'s emission
// shape are two documents that must agree, and they agree in one language.
type SubAxis struct {
	Axis   int    `json:"axis"`
	Extent string `json:"extent"`
}

// LayoutMorphism is one validated storage arrangement.
type LayoutMorphism struct {
	handle      Handle
	class       LayoutClass
	description string
	// rank pins the logical rank this arrangement is defined for. 0 means any
	// rank, which only the identity can honestly claim.
	rank    int
	subAxes []SubAxis
	extents []*shapeExpr
	// perm is the STORAGE order of the sub-axis list: perm[j] is the index of
	// the sub-axis that varies at storage position j.
	perm []int
	// candidate marks a wish emitted by a compiler that has not been ratified.
	// A candidate is parsed, validated and verified like any other record and
	// still NEVER derives: machines derive along ratified morphisms, they do
	// not invent them.
	candidate  bool
	provenance []string
}

func (m *LayoutMorphism) Handle() Handle       { return m.handle }
func (m *LayoutMorphism) Class() LayoutClass   { return m.class }
func (m *LayoutMorphism) Description() string  { return m.description }
func (m *LayoutMorphism) Rank() int            { return m.rank }
func (m *LayoutMorphism) SubAxes() []SubAxis   { return m.subAxes }
func (m *LayoutMorphism) Permutation() []int   { return append([]int(nil), m.perm...) }
func (m *LayoutMorphism) Candidate() bool      { return m.candidate }
func (m *LayoutMorphism) Ratified() bool       { return !m.candidate }
func (m *LayoutMorphism) Provenance() []string { return m.provenance }
func (m *LayoutMorphism) Digest() string       { return digestOf(m.canonical()) }
func (m *LayoutMorphism) Identity() bool       { return len(m.subAxes) == 0 }
func (m *LayoutMorphism) String() string       { return m.handle.String() }

func (m *LayoutMorphism) canonical() string {
	var out strings.Builder
	out.WriteString(LayoutMorphismFormat)
	fmt.Fprintf(&out, "\nname=%s\nversion=%d\nclass=%s\nrank=%d\ncandidate=%t\n",
		m.handle.Name, m.handle.Version, m.class, m.rank, m.candidate)
	for _, sub := range m.subAxes {
		fmt.Fprintf(&out, "sub axis=%d extent=%s\n", sub.Axis, sub.Extent)
	}
	rendered := make([]string, 0, len(m.perm))
	for _, at := range m.perm {
		rendered = append(rendered, fmt.Sprint(at))
	}
	fmt.Fprintf(&out, "perm=%s\n", strings.Join(rendered, ","))
	return out.String()
}

type rawLayoutMorphism struct {
	Format      *string   `json:"format"`
	Name        *string   `json:"name"`
	Version     *uint32   `json:"version"`
	Class       string    `json:"class"`
	Description string    `json:"description"`
	Rank        int       `json:"rank,omitempty"`
	SubAxes     []SubAxis `json:"sub_axes,omitempty"`
	Permutation []int     `json:"permutation,omitempty"`
	Candidate   bool      `json:"candidate,omitempty"`
	Provenance  []string  `json:"provenance"`
	Digest      string    `json:"digest,omitempty"`
}

// ParseLayoutMorphism reads and validates one layout record.
//
// Everything checkable WITHOUT a shape is checked here; everything that needs
// one is checked by Plan. The split matters: a record is loaded once at start
// and planned per tensor, so a malformed permutation must not wait for a
// checkpoint to arrive before it refuses.
func ParseLayoutMorphism(document []byte) (*LayoutMorphism, error) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.DisallowUnknownFields()
	var raw rawLayoutMorphism
	if err := decoder.Decode(&raw); err != nil {
		return nil, refuse("json", "%v", err)
	}
	if raw.Format == nil || *raw.Format != LayoutMorphismFormat {
		return nil, refuse("format", "not a %s document", LayoutMorphismFormat)
	}
	if raw.Name == nil || raw.Version == nil {
		return nil, refuse("identity", "a layout morphism is named and versioned")
	}
	handle, err := ParseHandle(fmt.Sprintf("%s@%d", *raw.Name, *raw.Version))
	if err != nil {
		return nil, err
	}
	class := LayoutClass(raw.Class)
	if class != ClassInductor && class != ClassEndpointDeclared {
		return nil, refuse("layout-class",
			"%s declares class %q; a layout is either the compiler's (%q) or a "+
				"kernel library's (%q)", handle, raw.Class, ClassInductor, ClassEndpointDeclared)
	}
	if len(raw.Provenance) == 0 {
		return nil, refuse("provenance",
			"%s carries no evidence. A layout is transcribed from the code that "+
				"reads these bytes; the citation is the only reason to trust it", handle)
	}
	morphism := &LayoutMorphism{
		handle: handle, class: class, description: raw.Description, rank: raw.Rank,
		subAxes: raw.SubAxes, perm: raw.Permutation, candidate: raw.Candidate,
		provenance: raw.Provenance,
	}
	if err := morphism.validate(); err != nil {
		return nil, err
	}
	if raw.Digest != "" && raw.Digest != morphism.Digest() {
		return nil, refuse("digest", "%s carries digest %s but hashes to %s",
			handle, raw.Digest, morphism.Digest())
	}
	return morphism, nil
}

func (m *LayoutMorphism) validate() error {
	if len(m.subAxes) == 0 {
		// The identity. It is a real record — it is the coordinate every
		// existing stamp carries — and it is the ONLY one that may be rankless.
		if len(m.perm) != 0 {
			return refuse("layout", "%s permutes a factorization it does not declare", m.handle)
		}
		if m.rank != 0 {
			return refuse("layout",
				"%s factors nothing, which is the identity for EVERY rank; pinning "+
					"it to rank %d would refuse tensors it arranges correctly",
				m.handle, m.rank)
		}
		return nil
	}
	if m.rank < 1 {
		return refuse("layout",
			"%s factors %d sub-axes without declaring the rank they factor",
			m.handle, len(m.subAxes))
	}
	if len(m.perm) != len(m.subAxes) {
		return refuse("layout", "%s permutes %d positions over %d sub-axes",
			m.handle, len(m.perm), len(m.subAxes))
	}
	// Sub-axes are grouped by axis, axes ascending, every axis covered. The
	// order WITHIN a group is the row-major order of the logical axis, so it is
	// load-bearing and cannot be sorted here.
	axis, covered := -1, 0
	for _, sub := range m.subAxes {
		switch sub.Axis {
		case axis: // another factor of the axis already open
		case axis + 1:
			axis, covered = sub.Axis, covered+1
		default:
			return refuse("layout",
				"%s lists axis %d after axis %d; sub-axes are grouped by axis, "+
					"outermost first, and no axis is skipped", m.handle, sub.Axis, axis)
		}
		expression, err := parseShapeExpr(sub.Extent)
		if err != nil {
			return refuse("layout", "%s: axis %d: %v", m.handle, sub.Axis, err)
		}
		m.extents = append(m.extents, expression)
	}
	if covered != m.rank {
		return refuse("layout", "%s declares rank %d and factors %d axes",
			m.handle, m.rank, covered)
	}
	seen := make([]bool, len(m.perm))
	for _, at := range m.perm {
		if at < 0 || at >= len(m.perm) {
			return refuse("layout", "%s permutes to position %d, which does not exist",
				m.handle, at)
		}
		if seen[at] {
			return refuse("layout",
				"%s sends two sub-axes to storage position %d. A layout that is not "+
					"a bijection loses elements, and it loses them silently", m.handle, at)
		}
		seen[at] = true
	}
	if m.changesNothing() {
		return refuse("layout",
			"%s is the identity for every shape — it factors the logical axes, "+
				"permutes nothing and pads nothing. Storing it as a second name for "+
				"%s would make one arrangement two identities", m.handle, DefaultLayout)
	}
	return nil
}

// changesNothing reports a record that is the identity map at EVERY shape:
// storage order equals declaration order and no extent can round up.
func (m *LayoutMorphism) changesNothing() bool {
	for at, to := range m.perm {
		if at != to {
			return false
		}
	}
	for _, sub := range m.subAxes {
		if strings.Contains(sub.Extent, "ceil(") {
			return false
		}
	}
	return true
}

// --- planning one arrangement over one concrete shape -----------------------

// LayoutPlan is what a layout morphism MEANS for one tensor: the storage
// extents, in storage order, and how many elements the destination holds.
//
// It is a decision, not a transform. The Rust applier derives the same extents
// from the same record and moves the bytes; nothing here writes one.
type LayoutPlan struct {
	// Source is the logical shape this plan was computed for.
	Source Shape
	// Extents are the sub-axis extents in DECLARATION order.
	Extents []uint64
	// Storage are the same extents in STORAGE order (extents[perm[j]]).
	Storage []uint64
	// Elements is the destination element count: the product of Storage.
	Elements uint64
	// Padded reports that the destination is LARGER than the source, because
	// some axis rounded up. A padded layout is injective, not bijective: it
	// round-trips on its image and the fill bytes are zero and meaningless.
	Padded bool
}

// Plan computes this arrangement for a concrete logical shape, or refuses.
func (m *LayoutMorphism) Plan(shape Shape) (*LayoutPlan, error) {
	if m.rank != 0 && len(shape) != m.rank {
		return nil, refuse("layout-rank",
			"%s arranges rank-%d tensors; this one is rank %d %s",
			m.handle, m.rank, len(shape), shape)
	}
	plan := &LayoutPlan{Source: shape.clone(), Elements: 1}
	if m.Identity() {
		plan.Extents = shape.clone()
		plan.Storage = shape.clone()
		for _, dimension := range shape {
			plan.Elements *= dimension
		}
		return plan, nil
	}
	product := uint64(1)
	axis := m.subAxes[0].Axis
	for at, sub := range m.subAxes {
		if sub.Axis != axis {
			if err := m.closeAxis(axis, product, shape); err != nil {
				return nil, err
			}
			if product > shape[axis] {
				plan.Padded = true
			}
			axis, product = sub.Axis, 1
		}
		extent, err := m.extents[at].eval(shape)
		if err != nil {
			return nil, refuse("layout", "%s: axis %d: %v", m.handle, sub.Axis, err)
		}
		if extent == 0 {
			return nil, refuse("layout", "%s: axis %d has a zero-sized factor %q",
				m.handle, sub.Axis, sub.Extent)
		}
		product *= extent
		plan.Extents = append(plan.Extents, extent)
		plan.Elements *= extent
	}
	if err := m.closeAxis(axis, product, shape); err != nil {
		return nil, err
	}
	if product > shape[axis] {
		plan.Padded = true
	}
	for _, at := range m.perm {
		plan.Storage = append(plan.Storage, plan.Extents[at])
	}
	return plan, nil
}

// closeAxis is the arithmetic that makes a factorization TRUE rather than
// plausible: the factors of an axis multiply to its dim, or they overshoot it
// by a declared rounding. Undershooting is the silent one — a layout that
// addresses 3968 of 4000 rows reads 32 rows of another tensor.
func (m *LayoutMorphism) closeAxis(axis int, product uint64, shape Shape) error {
	if axis >= len(shape) {
		return refuse("layout-rank", "%s factors axis %d of a rank-%d tensor",
			m.handle, axis, len(shape))
	}
	if product < shape[axis] {
		return refuse("layout",
			"%s factors axis %d into %d elements, and the axis is %d. A "+
				"factorization that does not reach its axis drops the tail",
			m.handle, axis, product, shape[axis])
	}
	return nil
}

// --- bridging two arrangements ---------------------------------------------

// Bridge is the identity of the map that takes bytes arranged as `from` and
// writes them arranged as `to`.
//
// It is COMPUTED, never authored. Each record is already the map from plain
// row-major; the bridge between any two of them is one record's inverse
// followed by the other's, so a hand-written A->B document would be a second
// place to write down a fact both records already carry. The rendering is what
// a derived tree's manifest names and what its derived digest is taken over,
// so it is an identity and it is stable.
func Bridge(from, to Handle) string {
	return from.String() + ">" + to.String()
}

// ParseBridge reads a bridge id back.
func ParseBridge(text string) (Handle, Handle, error) {
	left, right, found := strings.Cut(text, ">")
	if !found {
		return Handle{}, Handle{}, refuse("bridge", "%q is not <from>><to>", text)
	}
	fromHandle, err := ParseHandle(left)
	if err != nil {
		return Handle{}, Handle{}, err
	}
	toHandle, err := ParseHandle(right)
	if err != nil {
		return Handle{}, Handle{}, err
	}
	return fromHandle, toHandle, nil
}
