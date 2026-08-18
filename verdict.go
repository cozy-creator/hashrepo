package tensorfs

// The ARTIFACT-level verdict — the answer a bind gate asks: given this
// checkpoint's headers and this lane's contract, may the checkpoint bind?
//
// Three answers and no more, because tensorfs#122's matching-set ruling
// (Paul, 2026-08-18: "the layout + config + anything else shouldn't
// over-constrain, or under-constrain what checkpoints can run") makes the
// middle one mandatory. Two answers would force every fp16-packaged SDXL — the
// dominant community packaging, which the pipeline runs natively — into a flat
// refusal, and a flat refusal for a checkpoint that only needs a cast IS the
// over-constraint the ruling forbids.
//
// What this file does NOT decide, deliberately:
//
//   - CONFIG compatibility. A contract is a weights-LAYOUT template; whether
//     the author's pipeline class can be constructed from the checkpoint's own
//     config tree is a different predicate, answered by running it (th#2160's
//     config-only dry-load). Satisfies here means "the bytes are the right
//     shape", never "it will run".
//   - QUALITY. Ruled out of scope. SDXL-inpainting's 9-channel `conv_in` is
//     unclaimed by the SDXL contract, so it Satisfies, binds, runs to
//     completion, and serves garbage. That is a LEGAL admit.

import (
	"fmt"
	"sort"
	"strings"
)

// ArtifactFile is one tensor-carrying member of a checkpoint, with the header
// entries read out of it. Matching is PER FILE: a multifolder lane document is
// all-optional declarations, and each component file matches the families it
// carries.
type ArtifactFile struct {
	Path    string
	Tensors []InventoryTensor
}

// VerdictKind is the three-way answer.
type VerdictKind string

const (
	// VerdictSatisfies — the headers match the contract as they are.
	VerdictSatisfies VerdictKind = "satisfies"
	// VerdictDerivable — the headers do not match, but the only gap is one a
	// named conversion closes. The caller's obligation is to OFFER that
	// conversion, not to refuse.
	VerdictDerivable VerdictKind = "derivable"
	// VerdictIncompatible — a structural disagreement no conversion closes.
	VerdictIncompatible VerdictKind = "incompatible"
)

// Conversion names the work that would make a Derivable artifact satisfy.
type Conversion struct {
	// Kind is a stable token a caller may switch on. Today: "dtype-cast".
	Kind string
	// From are the dtypes observed, To the dtypes declared.
	From []string
	To   []string
	// Files are the artifact members that need the conversion.
	Files []string
}

func (c *Conversion) String() string {
	return fmt.Sprintf("%s from %s to %s (%d file(s): %s)",
		c.Kind, strings.Join(c.From, "|"), strings.Join(c.To, "|"),
		len(c.Files), strings.Join(c.Files, ", "))
}

// Verdict is the artifact-level answer.
type Verdict struct {
	Kind  VerdictKind
	Stamp string
	// Matched counts tensors explained across every member; Explained counts
	// the members the contract described.
	Matched   int
	Explained int
	// Unexplained are members the contract claimed no tensor of. NOT a
	// refusal on its own — a checkpoint may carry members outside a lane's
	// scope — but it is the evidence behind an incompatible verdict whose
	// every member was unexplained.
	Unexplained []string
	// Conversion is set iff Kind is VerdictDerivable.
	Conversion *Conversion
	// Mismatch is set iff Kind is VerdictIncompatible or VerdictDerivable: the
	// first disagreement, NAMING the tensor and the pattern.
	Mismatch *Mismatch
}

func (v Verdict) String() string {
	switch v.Kind {
	case VerdictSatisfies:
		return fmt.Sprintf("satisfies %s (%d tensors across %d file(s))", v.Stamp, v.Matched, v.Explained)
	case VerdictDerivable:
		return fmt.Sprintf("derivable to %s via %s — %s", v.Stamp, v.Conversion, v.Mismatch)
	default:
		return fmt.Sprintf("incompatible with %s — %s", v.Stamp, v.Mismatch)
	}
}

// Verdict answers the bind question for a whole artifact.
//
// The rule, in the order it must be applied:
//
//  1. A member with a STRUCTURAL disagreement (rank, fusion, permute, a missing
//     required declaration) is incompatible outright — no cast makes a
//     differently-shaped tensor the right tensor. Reported first so a
//     conversion is never offered for work that would not help.
//  2. Otherwise a member whose only disagreement is DTYPE makes the artifact
//     derivable. The lane means a particular packaging; a cast is math, so the
//     remedy is an ingest-time conversion job and never a load-time repair
//     (tensorfs#122: `adapter::decide` returns Conversion, and the streaming
//     loader that meets a lane-disagreeing dtype loads it verbatim with a
//     warning — "reported, never repaired").
//  3. Otherwise, if at least one member was explained, the artifact satisfies.
//  4. Otherwise nothing in the artifact is this contract's: incompatible.
//
// Members are visited in path order so the named tensor in a verdict is stable
// across calls — a refusal that names a different tensor each time reads as
// flakiness.
func (c *Contract) Verdict(files []ArtifactFile) Verdict {
	ordered := make([]ArtifactFile, len(files))
	copy(ordered, files)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Path < ordered[j].Path })

	verdict := Verdict{Stamp: c.Stamp()}
	var structural *Mismatch
	var dtypeFirst *Mismatch
	fromSet := map[string]struct{}{}
	toSet := map[string]struct{}{}
	var castFiles []string

	for _, file := range ordered {
		match, mismatch := c.Match(file.Tensors)
		switch {
		case mismatch == nil:
			verdict.Matched += match.Matched
			verdict.Explained++
		case mismatch.Kind == MismatchNothingExplained:
			verdict.Unexplained = append(verdict.Unexplained, file.Path)
		case mismatch.Kind == MismatchDtype:
			if dtypeFirst == nil {
				dtypeFirst = mismatch
			}
			fromSet[mismatch.Observed] = struct{}{}
			for _, declared := range strings.Split(mismatch.Declared, "|") {
				toSet[declared] = struct{}{}
			}
			castFiles = append(castFiles, file.Path)
		default:
			if structural == nil {
				structural = mismatch
			}
		}
	}

	switch {
	case structural != nil:
		verdict.Kind, verdict.Mismatch = VerdictIncompatible, structural
	case dtypeFirst != nil:
		verdict.Kind, verdict.Mismatch = VerdictDerivable, dtypeFirst
		verdict.Conversion = &Conversion{
			Kind: "dtype-cast", From: sortedKeys(fromSet), To: sortedKeys(toSet), Files: castFiles,
		}
	case verdict.Explained > 0:
		verdict.Kind = VerdictSatisfies
	default:
		verdict.Kind = VerdictIncompatible
		verdict.Mismatch = &Mismatch{Kind: MismatchNothingExplained, Stamp: verdict.Stamp}
	}
	return verdict
}

func sortedKeys(set map[string]struct{}) []string {
	out := make([]string, 0, len(set))
	for key := range set {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}
