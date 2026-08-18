package tensorfs

// The DERIVABILITY VERDICT: does this header set satisfy this contract, and if
// not, is the gap a CONVERSION or a real incompatibility?
//
// tensorfs#114 settled tensorhub's role as store+forward and named this as the
// explicit follow-up: "If a hub-side ahead-of-time derivability verdict is ever
// wanted, it is a tensorfs-go call over two stored documents." tensorfs#122's
// matching-set ruling made it wanted — the hub's bind gate (th#2160) must
// refuse a checkpoint the pipeline cannot run, and must NOT refuse one it can.
//
// Two properties this file exists to hold:
//
//   - The match rule is Contract::matches (contract.rs:976-1028) ported rule for
//     rule, not approximated. A Go answer that disagrees with the Rust one shows
//     up as a wrong ADMIT — a bind that should have been refused — which stays
//     invisible until a pod 500s. The port keeps the same order of operations:
//     first-declaration-wins, unclaimed tensors skipped, a claimed disagreement
//     refusing the WHOLE contract, required-then-floor at the end.
//   - A refusal NAMES the tensor and the pattern it failed on. "Incompatible"
//     with no name is not actionable by an operator and cannot be surfaced to a
//     user, so Mismatch carries both plus what was declared and what was
//     observed.

import (
	"fmt"
	"strings"
)

// InventoryTensor is one header entry, reduced to what a contract check needs:
// the planner's InventoryTensor without the offset.
//
// Dtype is the container's own spelling — safetensors' ("BF16", "F16", "F32")
// or the ggml type name — because that is what the contract's `dtypes` lists.
// Shape is LOGICAL row-major, outermost axis first (GGUF `ne` reversed), the
// same convention the Rust inventory carries.
type InventoryTensor struct {
	Name  string
	Dtype string
	Shape []uint64

	// Length is the tensor's byte extent. It is read ONLY by the fusion
	// divisibility check, and only that check.
	//
	// UNKNOWN IS EXPRESSIBLE, AND IT IS PERMISSIVE. A caller reading a
	// safetensors header has the extent; a caller holding a reduced inventory
	// may not. Zero means "not supplied", and the byte half of the fusion rule
	// is then SKIPPED while the shape half still applies. That is a deliberate
	// divergence from Rust in the only safe direction: a bind gate that
	// invented a refusal from an absent number would refuse checkpoints the
	// pipeline runs, which is exactly the over-constraint tensorfs#122 forbids.
	Length uint64
}

// Match is a successful contract match over one file.
type Match struct {
	// Stamp is the matched contract's identity — `name@version` for a library
	// document, `sha256:<hex>` for a custom.
	Stamp string
	// Matched counts TENSORS EXPLAINED, not declarations satisfied: several
	// tensors hit one `{i}` declaration and each is counted.
	Matched int
}

// MismatchKind is why a contract did not describe a file. The labels are stable
// and are meant to reach a user.
type MismatchKind string

const (
	// MismatchDtype is a claimed tensor whose dtype is outside the declared
	// list. THE ONE KIND THAT IS ROUTINELY A PACKAGING DIFFERENCE RATHER THAN A
	// DIFFERENT MODEL — see Verdict.
	MismatchDtype MismatchKind = "dtype"
	// MismatchRank is a claimed tensor with the wrong number of axes. A
	// different rank under the same key spelling is a different tensor.
	MismatchRank MismatchKind = "rank"
	// MismatchFusion is a declared seam whose arithmetic does not divide.
	MismatchFusion MismatchKind = "fusion"
	// MismatchPermute is a declared view that cannot resolve against the shape.
	MismatchPermute MismatchKind = "permute"
	// MismatchRequired is a required declaration no tensor satisfied.
	MismatchRequired MismatchKind = "required"
	// MismatchNothingExplained is the floor: the contract claimed no tensor of
	// this file at all. Without it every all-optional document — the
	// per-component-file shape a multifolder lane needs — would vacuously match
	// every tensor container in existence.
	MismatchNothingExplained MismatchKind = "nothing-explained"
)

// Mismatch is a NAMED refusal. Tensor and Pattern are the whole point: a
// verdict an operator cannot act on is not a verdict.
type Mismatch struct {
	Kind MismatchKind
	// Tensor is the header entry that failed. Empty only for
	// MismatchRequired (nothing was there to name) and
	// MismatchNothingExplained (the failure is about the file, not a tensor).
	Tensor string
	// Pattern is the declaration's key pattern; Role is its role pattern.
	// Both empty for MismatchNothingExplained.
	Pattern string
	Role    string
	// Declared renders what the contract asked for, Observed what the header
	// carried — the two halves a human compares.
	Declared string
	Observed string
	// Stamp is the contract that refused.
	Stamp string
}

func (m *Mismatch) Error() string { return m.String() }

func (m *Mismatch) String() string {
	switch m.Kind {
	case MismatchNothingExplained:
		return fmt.Sprintf("%s explains no tensor of this file", m.Stamp)
	case MismatchRequired:
		return fmt.Sprintf("%s requires %q, which no tensor satisfies", m.Stamp, m.Pattern)
	default:
		return fmt.Sprintf("%s: tensor %q matches pattern %q but its %s is %s, not %s",
			m.Stamp, m.Tensor, m.Pattern, m.Kind, m.Observed, m.Declared)
	}
}

// Tensors reports the contract's declarations in document order, as an opaque
// count — the accessors below index into it. It exists so a caller can size a
// per-declaration bookkeeping slice without the package exporting its internals.
func (c *Contract) Tensors() int { return len(c.tensors) }

// Match answers whether this contract describes the given file, and names the
// first disagreement when it does not.
//
// Exactly one of the two results is non-nil. This is Contract::matches
// (contract.rs:976-1028) with the refusal reason kept instead of collapsed to
// None: Rust returns Option because its callers only ask yes/no, and the whole
// value of a bind-time verdict is the reason.
func (c *Contract) Match(inventory []InventoryTensor) (*Match, *Mismatch) {
	stamp := c.Stamp()
	seen := make([]bool, len(c.tensors))
	matched := 0
	for _, tensor := range inventory {
		// FIRST declaration wins, in document order — the same rule as Rust's
		// `find`. Declaration order is therefore part of a contract's meaning.
		index := -1
		for i := range c.tensors {
			if c.tensors[i].pattern.matches(tensor.Name) {
				index = i
				break
			}
		}
		if index < 0 {
			// An unclaimed tensor is neither a refusal nor a credit. A file may
			// carry arbitrary extra tensors — which is precisely why the SDXL
			// contract never seeing unet `conv_in` lets a 9-channel inpainting
			// checkpoint through (tensorfs#122; ACCEPTED, it runs to completion).
			continue
		}
		entry := &c.tensors[index]
		if bad := entry.accepts(stamp, tensor); bad != nil {
			return nil, bad
		}
		seen[index] = true
		matched++
	}
	for i := range c.tensors {
		if c.tensors[i].required && !seen[i] {
			return nil, &Mismatch{
				Kind: MismatchRequired, Stamp: stamp,
				Pattern: c.tensors[i].pattern.text, Role: c.tensors[i].role.text,
			}
		}
	}
	if matched == 0 {
		return nil, &Mismatch{Kind: MismatchNothingExplained, Stamp: stamp}
	}
	return &Match{Stamp: stamp, Matched: matched}, nil
}

// accepts is the per-tensor half: dtype, rank, then the seam arithmetic. A
// claimed tensor that disagrees is a refusal, not a shrug — contracts must be
// falsifiable from the header alone or the stamp would mean nothing.
func (t *contractTensor) accepts(stamp string, tensor InventoryTensor) *Mismatch {
	bad := func(kind MismatchKind, declared, observed string) *Mismatch {
		return &Mismatch{
			Kind: kind, Stamp: stamp, Tensor: tensor.Name,
			Pattern: t.pattern.text, Role: t.role.text,
			Declared: declared, Observed: observed,
		}
	}
	if len(t.dtypes) > 0 {
		found := false
		for _, declared := range t.dtypes {
			if declared == tensor.Dtype {
				found = true
				break
			}
		}
		if !found {
			return bad(MismatchDtype, strings.Join(t.dtypes, "|"), tensor.Dtype)
		}
	}
	if t.rank != nil && *t.rank != uint64(len(tensor.Shape)) {
		return bad(MismatchRank, fmt.Sprintf("%d", *t.rank), fmt.Sprintf("%d", len(tensor.Shape)))
	}
	if t.fusion != nil && !t.fusion.divides(tensor.Shape, tensor.Length) {
		return bad(MismatchFusion, t.fusion.describe(), shapeText(tensor.Shape))
	}
	if t.permute != nil && !t.permute.resolves(tensor.Shape) {
		return bad(MismatchPermute, "a resolvable view", shapeText(tensor.Shape))
	}
	return nil
}

// divides is Fusion::runs reduced to its yes/no: the outer axis must divide
// into the declared units, and the extent must be a whole number of rows.
//
// The 1 MiB seam floor (MIN_SEAM_PART_BYTES) is deliberately NOT applied — Rust
// applies it in runs_of/seam_offsets, never in matches. A fusion whose runs fall
// below it still MATCHES; it just yields no cut points.
func (f *contractFusion) divides(shape []uint64, length uint64) bool {
	if len(shape) == 0 {
		return false
	}
	rows := shape[0]
	units := uint64(0)
	for _, part := range f.parts {
		units += part.share
	}
	units *= f.groups
	if rows == 0 || units == 0 || rows%units != 0 {
		return false
	}
	// Length 0 is "not supplied" — see InventoryTensor.Length. Skipping the
	// byte half here is the permissive divergence, never a manufactured refusal.
	if length != 0 && length%rows != 0 {
		return false
	}
	return true
}

func (f *contractFusion) describe() string {
	parts := make([]string, 0, len(f.parts))
	for _, part := range f.parts {
		role := part.role
		if role == "" {
			role = "-"
		}
		parts = append(parts, fmt.Sprintf("%s:%d", role, part.share))
	}
	return fmt.Sprintf("an outer axis divisible by %d group(s) of [%s]", f.groups, strings.Join(parts, " "))
}

// resolves is Permute::resolve reduced to its yes/no.
func (p *contractPermute) resolves(shape []uint64) bool {
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension != 0 && elements > (1<<62)/dimension {
			return false
		}
		elements *= dimension
	}
	known := uint64(1)
	autoSeen := false
	for _, dimension := range p.view {
		var value uint64
		switch dimension.kind {
		case dimLiteral:
			value = dimension.literal
		case dimAxis:
			if dimension.axis >= uint64(len(shape)) {
				return false
			}
			axis := shape[dimension.axis]
			if dimension.divisor == 0 || axis%dimension.divisor != 0 {
				return false
			}
			value = axis / dimension.divisor
		default: // dimAuto
			if autoSeen {
				return false
			}
			autoSeen = true
			value = 1
		}
		if value == 0 {
			return false
		}
		if known > (1<<62)/value {
			return false
		}
		known *= value
	}
	if !autoSeen {
		return known == elements
	}
	return known != 0 && elements%known == 0
}

// matches reports whether one tensor name is an instance of this pattern.
//
// Literal pieces are consumed left to right and each {i} takes a greedy run of
// ASCII digits. Leading zeros are refused — "layers.01.weight" is not an
// instance of "layers.{i}.weight" — so one tensor name has one spelling and a
// document cannot claim the same tensor twice through two renderings.
func (p contractPattern) matches(name string) bool {
	rest := name
	text := p.text
	for {
		at := strings.Index(text, "{i}")
		if at < 0 {
			return rest == text
		}
		literal := text[:at]
		if !strings.HasPrefix(rest, literal) {
			return false
		}
		rest = rest[len(literal):]
		digits := 0
		for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
			digits++
		}
		if digits == 0 || (digits > 1 && rest[0] == '0') {
			return false
		}
		rest = rest[digits:]
		text = text[at+3:]
	}
}

func shapeText(shape []uint64) string {
	parts := make([]string, 0, len(shape))
	for _, dimension := range shape {
		parts = append(parts, fmt.Sprintf("%d", dimension))
	}
	return "[" + strings.Join(parts, ",") + "]"
}
