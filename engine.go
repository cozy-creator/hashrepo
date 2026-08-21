package tensorfs

// THE DECISION ENGINE. One implementation, in one language.
//
// v1 had three matchers — Rust `Contract::matches`, its Go port, and the
// conversion planner's Python claim rule — and three copies of a rule that
// decides an ADMIT are three chances to admit a bind that should have been
// refused, invisible until a pod 500s. This is the replacement, and the
// verdict stays the hub's: workers consume binding-carried verdicts.
//
// Two entry points:
//
//	Stamp(headers)         -> (topology, quant), by searching the catalog for
//	                          the (topology, rule) pair whose COMPUTED layout
//	                          the headers are. This is "inverting the rule out
//	                          of the header" made literal.
//	Admit(headers, lanes)  -> ADMIT | DERIVABLE(job) | REFUSE(named tensor)
//
// FAIL-CLOSED, on purpose. v1's `Verdict` admitted whenever at least one member
// was explained and nothing contradicted, so `sd15.diffusers-bf16@1` SATISFIED
// the ERNIE checkpoint on the strength of four shared keys, and a
// required:false-everywhere document admitted anything at all. Here a
// candidate must BE the layout: every non-optional key present, at the declared
// shape, with an accepted dtype, and nothing left over.

import (
	"fmt"
	"sort"
	"strings"
)

// RefusalReason is why admission said no. Stable labels — they reach a user.
type RefusalReason string

const (
	// RefusalShape — a tensor is there under the right name and the wrong
	// shape. THE ONE V1 COULD NOT SEE.
	RefusalShape RefusalReason = "shape"
	// RefusalDtype — right name, right shape, wrong element type.
	RefusalDtype RefusalReason = "dtype"
	// RefusalUnclaimed — the candidate carries a tensor the layout does not.
	RefusalUnclaimed RefusalReason = "unclaimed"
	// RefusalMissing — the layout declares a tensor the candidate lacks.
	RefusalMissing RefusalReason = "missing"
	// RefusalDuplicate — two members of one component carry the same key.
	RefusalDuplicate RefusalReason = "duplicate"
	// RefusalNoStamp — the headers are no catalogued (topology, quant) pair.
	RefusalNoStamp RefusalReason = "no-stamp"
	// RefusalUnrelated — a stamp that no declared lane relates to.
	RefusalUnrelated RefusalReason = "unrelated"
	// RefusalNoLane — the endpoint declared no lane at all.
	RefusalNoLane RefusalReason = "no-lanes"
	// RefusalNoBridge — the tensors and the element type line up and the
	// ARRANGEMENT does not: the lane wants its bytes laid out by a layout
	// morphism no ratified record bridges to. Named separately because the
	// operator's next move is a catalog entry, not a different checkpoint.
	RefusalNoBridge RefusalReason = "no-bridge"
)

// Refusal NAMES the disagreement. A verdict an operator cannot act on is not a
// verdict, and every field below exists because a refusal without it sent
// somebody to read 40 GB of weights to find out what a header already said.
type Refusal struct {
	Reason RefusalReason
	// Lane is what the candidate was measured against.
	Lane LayoutID
	// Member is the artifact file the disagreement is in — load-bearing,
	// because one contract routinely matches some members and refuses others.
	Member    string
	Component string
	Tensor    string
	Declared  string
	Observed  string
	// Matched counts what DID line up, which is how "close but wrong" is told
	// from "a different model entirely".
	Matched int
	Note    string
}

func (r *Refusal) Error() string { return r.String() }

func (r *Refusal) String() string {
	switch r.Reason {
	case RefusalNoLane:
		return "refused: the endpoint declares no lane"
	case RefusalNoStamp:
		return fmt.Sprintf("refused: these headers are no catalogued layout (%s)", r.Note)
	case RefusalMissing:
		return fmt.Sprintf("refused by %s: %s declares %q (%s), which the checkpoint "+
			"does not carry — %d of its tensors did line up%s",
			r.Lane, componentLabel(r.Component), r.Tensor, r.Declared, r.Matched,
			stampSuffix(r.Note))
	case RefusalUnclaimed:
		return fmt.Sprintf("refused by %s: %s carries %q (%s), which %s does not "+
			"declare — %d of its tensors did line up%s",
			r.Lane, memberLabel(r.Member), r.Tensor, r.Observed, r.Lane, r.Matched,
			stampSuffix(r.Note))
	case RefusalDuplicate:
		return fmt.Sprintf("refused by %s: %q appears in two members of %s",
			r.Lane, r.Tensor, componentLabel(r.Component))
	case RefusalUnrelated:
		return fmt.Sprintf("refused: %s is not %s and no morphism relates them", r.Note, r.Lane)
	case RefusalNoBridge:
		return fmt.Sprintf("refused by %s: the checkpoint's tensors and element type "+
			"reach this lane, and its BYTE ARRANGEMENT does not — stored as %s, the "+
			"lane declares %s, and %s", r.Lane, r.Observed, r.Declared, r.Note)
	default:
		return fmt.Sprintf("refused by %s: in %s, %q has %s %s, not %s%s",
			r.Lane, memberLabel(r.Member), r.Tensor, r.Reason, r.Observed, r.Declared,
			stampSuffix(r.Note))
	}
}

// stampSuffix appends what the candidate IS, when it is anything — the other
// half of an actionable refusal.
func stampSuffix(note string) string {
	if note == "" {
		return ""
	}
	return " (it stamps " + note + ")"
}

func memberLabel(member string) string {
	if member == "" {
		return "the checkpoint"
	}
	return member
}

func componentLabel(component string) string {
	if component == "" {
		return "the checkpoint"
	}
	return "component " + component
}

// --- matching one candidate against one computed layout --------------------

// match is what a successful layout match reports: which member went to which
// component, and how many entries lined up. The assignment is kept because the
// derivation planner needs to know which component a member's bytes belong to.
type match struct {
	assignment map[string]string // member path -> component name
	matched    int
}

// matchLayout answers whether these headers ARE this layout.
//
// Members are grouped into components by CONTENT, not by directory name.
// tensorhub repacks trees, and a decision that turned on a folder spelling
// would refuse the same bytes under a different name. The directory is a
// tie-break, never the authority.
func matchLayout(files []ArtifactFile, layout *Layout) (*match, *Refusal) {
	ordered := append([]ArtifactFile(nil), files...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Path < ordered[j].Path })

	result := &match{assignment: map[string]string{}}
	pending := make([][]int, len(ordered)) // member -> candidate component indexes
	for at := range ordered {
		for index := range layout.Components {
			if accepted, _ := componentAccepts(&layout.Components[index], ordered[at]); accepted {
				pending[at] = append(pending[at], index)
			}
		}
		if len(pending[at]) == 0 {
			return nil, bestRefusal(layout, ordered[at])
		}
	}
	covered := make([]map[string]bool, len(layout.Components))
	for index := range covered {
		covered[index] = map[string]bool{}
	}
	assign := func(at, index int) *Refusal {
		component := &layout.Components[index]
		for _, tensor := range ordered[at].Tensors {
			if covered[index][tensor.Name] {
				return &Refusal{
					Reason: RefusalDuplicate, Lane: layout.ID, Member: ordered[at].Path,
					Component: component.Name, Tensor: tensor.Name,
				}
			}
			covered[index][tensor.Name] = true
			result.matched++
		}
		result.assignment[ordered[at].Path] = component.Name
		return nil
	}
	// Unambiguous members first, so an ambiguous one is resolved against what
	// is actually left rather than against the whole component.
	ambiguous := []int{}
	for at := range ordered {
		if len(pending[at]) == 1 {
			if refusal := assign(at, pending[at][0]); refusal != nil {
				return nil, refusal
			}
			continue
		}
		ambiguous = append(ambiguous, at)
	}
	for _, at := range ambiguous {
		best, bestScore := -1, -1
		for _, index := range pending[at] {
			score := 0
			for _, tensor := range ordered[at].Tensors {
				if !covered[index][tensor.Name] {
					score++
				}
			}
			// The directory name breaks a true tie: it is a hint the packaging
			// itself carries, and a deterministic answer beats a lucky one.
			if directoryHint(ordered[at].Path) == layout.Components[index].Name {
				score++
			}
			if score > bestScore {
				best, bestScore = index, score
			}
		}
		if refusal := assign(at, best); refusal != nil {
			return nil, refusal
		}
	}
	for index := range layout.Components {
		component := &layout.Components[index]
		for _, key := range component.keys {
			if covered[index][key] || component.Tensors[key].Optional {
				continue
			}
			return nil, &Refusal{
				Reason: RefusalMissing, Lane: layout.ID, Component: component.Name,
				Tensor: key, Declared: component.Tensors[key].String(),
				Matched: result.matched,
			}
		}
	}
	return result, nil
}

func directoryHint(memberPath string) string {
	at := strings.LastIndex(memberPath, "/")
	if at < 0 {
		return ""
	}
	rest := memberPath[:at]
	if slash := strings.LastIndex(rest, "/"); slash >= 0 {
		return rest[slash+1:]
	}
	return rest
}

// componentAccepts reports whether every tensor of one member is declared by
// one component, at its shape and an accepted dtype.
//
// The refusal's Matched is the number of entries that lined up BEFORE the
// disagreement; overlap() is the whole-member proximity used to choose which
// component's refusal to report. They are different questions and conflating
// them made a wrong-component refusal name a tensor at random.
func componentAccepts(component *LayoutComponent, file ArtifactFile) (bool, *Refusal) {
	matched := 0
	for _, tensor := range file.Tensors {
		declared, found := component.Tensors[tensor.Name]
		if !found {
			return false, &Refusal{
				Reason: RefusalUnclaimed, Member: file.Path, Component: component.Name,
				Tensor: tensor.Name, Observed: Shape(tensor.Shape).String() + " " + tensor.Dtype,
				Matched: matched,
			}
		}
		if !declared.Shape.Equal(Shape(tensor.Shape)) {
			return false, &Refusal{
				Reason: RefusalShape, Member: file.Path, Component: component.Name,
				Tensor: tensor.Name, Declared: declared.Shape.String(),
				Observed: Shape(tensor.Shape).String(), Matched: matched,
			}
		}
		if !declared.accepts(tensor.Dtype) {
			return false, &Refusal{
				Reason: RefusalDtype, Member: file.Path, Component: component.Name,
				Tensor: tensor.Name, Declared: strings.Join(declared.Dtypes, "|"),
				Observed: tensor.Dtype, Matched: matched,
			}
		}
		matched++
	}
	return true, nil
}

// overlap counts how many of a member's keys the component declares AT ALL —
// name only. It is the proximity measure, because "we share 1670 of 1680 key
// names and disagree on conv_in's shape" is the useful refusal and "we share
// nothing" is the useless one.
func overlap(component *LayoutComponent, file ArtifactFile) int {
	shared := 0
	for _, tensor := range file.Tensors {
		if _, found := component.Tensors[tensor.Name]; found {
			shared++
		}
	}
	return shared
}

// bestRefusal picks the component that came CLOSEST and reports its first
// disagreement. "This is not the layout" is unactionable; "conv_in.weight is
// [320, 9, 3, 3], not [320, 4, 3, 3]" is the whole answer.
func bestRefusal(layout *Layout, file ArtifactFile) *Refusal {
	var best *Refusal
	bestOverlap := -1
	for index := range layout.Components {
		_, refusal := componentAccepts(&layout.Components[index], file)
		if refusal == nil {
			continue
		}
		if shared := overlap(&layout.Components[index], file); shared > bestOverlap {
			best, bestOverlap = refusal, shared
		}
	}
	if best == nil {
		best = &Refusal{Reason: RefusalUnclaimed, Member: file.Path}
	}
	best.Lane = layout.ID
	return best
}

// --- stamping --------------------------------------------------------------

// Stamp identifies a checkpoint's bytes as a (topology, quant) pair.
//
// The search is over the catalog's (topology, rule) pairs in canonical order,
// and the FIRST layout the headers are is the answer. A candidate can be at
// most one: two pairs whose computed layouts are identical would be two names
// for one packaging, and the catalog's self-validation would rather say so.
func (c *Catalog) Stamp(files []ArtifactFile) (LayoutID, *Refusal) {
	total, probe := candidateShape(files)
	var closest *Refusal
	for _, id := range c.order {
		layout, plausible := c.plausible(id, total, probe)
		if !plausible {
			continue
		}
		if _, refusal := matchLayout(files, layout); refusal == nil {
			return id, nil
		} else if closest == nil || refusal.Matched > closest.Matched {
			closest = refusal
		}
	}
	if closest == nil {
		return LayoutID{}, &Refusal{
			Reason: RefusalNoStamp,
			Note: fmt.Sprintf("%d tensors across %d member(s) reach no catalogued "+
				"layout's entry count", total, len(files)),
		}
	}
	return LayoutID{}, closest
}

// declares reports whether any component carries this key name.
func (l *Layout) declares(key string) bool {
	for at := range l.Components {
		if _, found := l.Components[at].Tensors[key]; found {
			return true
		}
	}
	return false
}

func (l *Layout) counts() (required, optional int) {
	for at := range l.Components {
		for _, tensor := range l.Components[at].Tensors {
			if tensor.Optional {
				optional++
			} else {
				required++
			}
		}
	}
	return required, optional
}

// StampAll returns EVERY catalogued pair these headers are, not just the first.
//
// It exists to be zero at all times but one. `Stamp` takes the first match, so
// "first" must never be able to matter; a checkpoint that is two stamps is two
// different things on two days, and the ambiguity would be invisible from the
// answer alone. The corpus test runs this over every banked checkpoint.
func (c *Catalog) StampAll(files []ArtifactFile) []LayoutID {
	total, probe := candidateShape(files)
	var found []LayoutID
	for _, id := range c.order {
		layout, plausible := c.plausible(id, total, probe)
		if !plausible {
			continue
		}
		if _, refusal := matchLayout(files, layout); refusal == nil {
			found = append(found, id)
		}
	}
	return found
}

// candidateShape is what the prefilters need: how many entries the headers
// carry, and one arbitrary key name from them.
func candidateShape(files []ArtifactFile) (total int, probe string) {
	for _, file := range files {
		total += len(file.Tensors)
		for _, tensor := range file.Tensors {
			if probe == "" || tensor.Name < probe {
				probe = tensor.Name
			}
		}
	}
	return total, probe
}

// plausible discards a pair without walking its tensors.
//
// Both filters are SOUND — never a false negative — because a matching layout
// must declare every key the candidate carries: the entry count has to be
// reachable, and one arbitrary candidate key has to be declared somewhere. The
// third check is not a filter but a rule: a transforming rule that found
// nothing eligible in this topology computes its base plain rule's header, and
// stamping that would invent a quantization the bytes do not have.
func (c *Catalog) plausible(id LayoutID, total int, probe string) (*Layout, bool) {
	layout, err := c.Layout(id)
	if err != nil {
		return nil, false
	}
	if rule := c.rules[id.Quant]; rule.Transforms() && layout.Transformed() == 0 {
		return nil, false
	}
	required, optional := layout.counts()
	if total < required || total > required+optional {
		return nil, false
	}
	if probe != "" && !layout.declares(probe) {
		return nil, false
	}
	return layout, true
}

// --- admission -------------------------------------------------------------

// DecisionKind is the three-way answer, and there are three because a flat
// refusal for a checkpoint that only needs a cast is the over-constraint
// tensorfs#122's matching-set ruling forbids.
type DecisionKind string

const (
	// DecisionAdmit — the stamp is one the endpoint declares. Serve it.
	DecisionAdmit DecisionKind = "admit"
	// DecisionDerivable — a named, priced job closes the gap. The endpoint
	// serves the PRODUCED artifact, never the input.
	DecisionDerivable DecisionKind = "derivable"
	// DecisionRefuse — no job closes it. Named, and PRE-DOWNLOAD.
	DecisionRefuse DecisionKind = "refuse"
)

// StepKind names what one derivation step does.
type StepKind string

const (
	StepRekey      StepKind = "rekey"
	StepSeam       StepKind = "seam"
	StepRelayout   StepKind = "relayout"
	StepCast       StepKind = "cast"
	StepQuantize   StepKind = "quantize"
	StepDequantize StepKind = "dequantize"
	StepRequantize StepKind = "requantize"
)

// DerivationStep is one leg of a derivation, priced by its storage tier.
type DerivationStep struct {
	Kind     StepKind
	Tier     StorageTier
	Morphism Handle
	From, To Handle
	Detail   string
}

func (s DerivationStep) String() string {
	if s.Morphism.Name != "" {
		return fmt.Sprintf("%s %s -> %s via %s [%s]", s.Kind, s.From, s.To, s.Morphism, s.Tier)
	}
	return fmt.Sprintf("%s %s -> %s [%s]", s.Kind, s.From, s.To, s.Tier)
}

// Decision is the answer a bind gate acts on.
type Decision struct {
	Kind DecisionKind
	// Stamp is what the candidate IS, when it is anything.
	Stamp LayoutID
	// Lane is the declared lane the decision is about.
	Lane LayoutID
	// Steps is the derivation, cheapest first-and-only. Empty for an admit.
	Steps []DerivationStep
	// Refusal names the disagreement. Set iff Kind is DecisionRefuse.
	Refusal *Refusal
}

// Lossy reports whether any step of the derivation loses information — which
// is what decides whether the job is gated and its output stored.
func (d Decision) Lossy() bool {
	for _, step := range d.Steps {
		if step.Tier == TierQuant {
			return true
		}
	}
	return false
}

func (d Decision) String() string {
	switch d.Kind {
	case DecisionAdmit:
		return fmt.Sprintf("admit %s", d.Stamp)
	case DecisionDerivable:
		rendered := make([]string, 0, len(d.Steps))
		for _, step := range d.Steps {
			rendered = append(rendered, step.String())
		}
		return fmt.Sprintf("derivable %s -> %s: %s",
			d.Stamp, d.Lane, strings.Join(rendered, "; "))
	default:
		return d.Refusal.String()
	}
}

// Admit is THE decision procedure (design §1.7).
//
//  1. stamp the candidate from its headers — catalog search including morphism
//     compositions;
//  2. match against the endpoint's declared lanes: the exact pair admits; the
//     same topology with a different quant is a conversion; a morphism-related
//     topology is a repack, then a conversion;
//  3. cheapest derivation wins.
//
// Every arm is decided from HEADERS ALONE, so a refusal happens before a byte
// of weights is fetched.
func (c *Catalog) Admit(files []ArtifactFile, lanes []LayoutID) Decision {
	if len(lanes) == 0 {
		return Decision{Kind: DecisionRefuse, Refusal: &Refusal{Reason: RefusalNoLane}}
	}
	stamp, refusal := c.Stamp(files)
	if refusal != nil {
		// Unstamped: measure against the declared lanes themselves, so the
		// refusal names a tensor of the lane the operator asked for rather
		// than of whatever the catalog came closest to.
		return Decision{Kind: DecisionRefuse, Refusal: c.closestLaneRefusal(files, lanes, refusal)}
	}
	for _, lane := range lanes {
		if stamp.Equal(lane) {
			return Decision{Kind: DecisionAdmit, Stamp: stamp, Lane: lane}
		}
	}
	var best *Decision
	for _, lane := range lanes {
		steps, ok := c.derive(stamp, lane)
		if !ok {
			continue
		}
		candidate := Decision{Kind: DecisionDerivable, Stamp: stamp, Lane: lane, Steps: steps}
		if best == nil || cost(candidate.Steps) < cost(best.Steps) {
			copied := candidate
			best = &copied
		}
	}
	if best != nil {
		return *best
	}
	// Before falling back to "these are different models", check whether the
	// ONLY thing in the way was the arrangement. That refusal has a different
	// remedy — a catalog entry, or a ratification — and saying "sdxl.diffusers
	// is not sdxl.diffusers" because a layout coordinate differs would send an
	// operator looking for a checkpoint problem that is not there.
	if refusal := c.bridgeRefusal(stamp, lanes); refusal != nil {
		return Decision{Kind: DecisionRefuse, Stamp: stamp, Lane: refusal.Lane, Refusal: refusal}
	}
	// A stamped candidate that no lane relates to still gets a NAMED tensor.
	// "sd15.diffusers@1 is not sd2.diffusers@1" tells an operator which two
	// things disagree; "conv_in.weight is [320, 9, 3, 3], not [320, 4, 3, 3]"
	// tells them WHY, and that is the sentence a refusal exists to produce.
	refusal = c.closestLaneRefusal(files, lanes, &Refusal{
		Reason: RefusalUnrelated, Lane: lanes[0], Note: stamp.String(),
	})
	refusal.Note = stamp.String()
	return Decision{Kind: DecisionRefuse, Stamp: stamp, Lane: refusal.Lane, Refusal: refusal}
}

// bridgeRefusal names the arrangement gap, for the first declared lane whose
// tensors and element type this stamp DOES reach.
func (c *Catalog) bridgeRefusal(stamp LayoutID, lanes []LayoutID) *Refusal {
	for _, lane := range lanes {
		if _, ok := c.repack(stamp.Topology, lane.Topology); !ok {
			continue
		}
		if _, ok := c.requant(stamp.Quant, lane.Quant); !ok {
			continue
		}
		_, why, ok := c.relayout(stamp.Bytes, lane.Bytes)
		if ok {
			continue
		}
		stored, declared := stamp.Normalized().Bytes, lane.Normalized().Bytes
		return &Refusal{
			Reason: RefusalNoBridge, Lane: lane, Note: why,
			Observed: stored.String(), Declared: declared.String(),
		}
	}
	return nil
}

func (c *Catalog) closestLaneRefusal(files []ArtifactFile, lanes []LayoutID, fallback *Refusal) *Refusal {
	var best *Refusal
	for _, lane := range lanes {
		layout, err := c.Layout(lane)
		if err != nil {
			continue
		}
		_, refusal := matchLayout(files, layout)
		if refusal == nil {
			continue // impossible: it would have stamped
		}
		if best == nil || refusal.Matched > best.Matched {
			best = refusal
		}
	}
	if best == nil {
		return fallback
	}
	return best
}

func cost(steps []DerivationStep) int {
	total := 0
	for _, step := range steps {
		total += int(step.Tier)
	}
	return total
}

// derive plans the cheapest route from a stamp to a lane, or reports that none
// exists. Topology first (a repack), then quant (a conversion) — the order the
// tiers price and the order a producer runs them in.
//
// ONE LIMIT, STATED RATHER THAN HIDDEN: a repack of an already-quantized
// checkpoint is planned here as if the seams applied to the stored elements.
// For a REKEY that is exactly true — names change, bytes do not. For a SEAM it
// is true only when the quantization's scale span survives the cut (a per-row
// fp8 scale does; a per-16-block one across a fused axis needs the same cut
// applied to the scale tensor). No such derivation is reachable in today's
// catalog — every seam relates two plain packagings — so the case is a
// refusal-by-absence rather than a wrong plan, and the producer that first
// needs it owes the scale arithmetic here.
func (c *Catalog) derive(stamp, lane LayoutID) ([]DerivationStep, bool) {
	steps, ok := c.repack(stamp.Topology, lane.Topology)
	if !ok {
		return nil, false
	}
	quantStep, ok := c.requant(stamp.Quant, lane.Quant)
	if !ok {
		return nil, false
	}
	if quantStep != nil {
		steps = append(steps, *quantStep)
	}
	// The arrangement leg runs LAST, and it has to: a quant conversion produces
	// new tensors, and it is those tensors' bytes the lane wants arranged its
	// way. Relayouting first would arrange bytes the conversion then replaces.
	layoutStep, _, ok := c.relayout(stamp.Bytes, lane.Bytes)
	if !ok {
		return nil, false
	}
	if layoutStep != nil {
		steps = append(steps, *layoutStep)
	}
	return steps, true
}

// relayout plans the byte-arrangement leg, or says WHY there is none.
//
// There is nothing to search. Every layout record is already the map from
// plain row-major, so the bridge from one to another is one record's inverse
// followed by the other's — computed, cheap, and impossible to write down
// inconsistently. What can fail is a coordinate the corpus does not carry, a
// candidate wish nothing has ratified, or two records whose ranks mean they
// can never describe one tensor.
func (c *Catalog) relayout(from, to Handle) (*DerivationStep, string, bool) {
	if from == (Handle{}) {
		from = DefaultLayout
	}
	if to == (Handle{}) {
		to = DefaultLayout
	}
	if from == to {
		return nil, "", true
	}
	source, target := c.arrangements[from], c.arrangements[to]
	// Checked in a fixed order — stored side first — so one bad pairing has one
	// refusal text, not whichever of two a map walk reached first.
	for _, side := range []struct {
		handle Handle
		record *LayoutMorphism
	}{{from, source}, {to, target}} {
		if side.record == nil {
			return nil, fmt.Sprintf("the corpus carries no %s record", side.handle), false
		}
		if side.record.Candidate() {
			return nil, fmt.Sprintf(
				"%s is an UNRATIFIED candidate — a wish a compiler emitted, which "+
					"nothing has ratified. Machines derive along ratified morphisms; "+
					"they do not invent them", side.handle), false
		}
	}
	if source.rank != 0 && target.rank != 0 && source.rank != target.rank {
		return nil, fmt.Sprintf(
			"%s arranges rank-%d tensors and %s arranges rank-%d ones, so no tensor "+
				"is described by both", from, source.rank, to, target.rank), false
	}
	// T3: the arrangement is either applied in flight on the load copy or
	// materialized once. Either way it is priced as work over the same
	// elements, never as new information.
	return &DerivationStep{
		Kind: StepRelayout, Tier: TierInterleaved, From: from, To: to,
		Detail: Bridge(from, to),
	}, "", true
}

// repack is a breadth-first search over the morphism catalog. Composition is
// the point: `net.*` -> `model.diffusion_model.*` -> a fused-qkv packaging is
// two ratified facts, not a third document nobody wrote.
func (c *Catalog) repack(from, to Handle) ([]DerivationStep, bool) {
	if from == to {
		return nil, true
	}
	type node struct {
		handle Handle
		steps  []DerivationStep
	}
	seen := map[Handle]bool{from: true}
	queue := []node{{handle: from}}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		outgoing := c.outgoing[current.handle]
		sorted := append([]*Morphism(nil), outgoing...)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].handle.Name < sorted[j].handle.Name
		})
		for _, morphism := range sorted {
			if seen[morphism.to] {
				continue
			}
			kind := StepRekey
			if len(morphism.seams) > 0 {
				kind = StepSeam
			}
			steps := append(append([]DerivationStep(nil), current.steps...), DerivationStep{
				Kind: kind, Tier: morphism.tier, Morphism: morphism.handle,
				From: morphism.from, To: morphism.to,
				Detail: morphism.description,
			})
			if morphism.to == to {
				return steps, true
			}
			seen[morphism.to] = true
			queue = append(queue, node{handle: morphism.to, steps: steps})
		}
	}
	return nil, false
}

// requant plans the quant leg, and REFUSES the ones no math closes: a rule
// that states no dequant equation cannot be read back, so nothing can be
// derived FROM it.
func (c *Catalog) requant(from, to Handle) (*DerivationStep, bool) {
	if from == to {
		return nil, true
	}
	source, target := c.rules[from], c.rules[to]
	if source == nil || target == nil {
		return nil, false
	}
	step := DerivationStep{Tier: TierQuant, From: from, To: to}
	switch {
	case !source.Transforms() && !target.Transforms():
		step.Kind = StepCast
		step.Detail = fmt.Sprintf("%s -> %s", source.DeclaredDtype, target.DeclaredDtype)
	case !source.Transforms():
		step.Kind = StepQuantize
		step.Detail = target.Inverse
	case !target.Transforms():
		if source.Inverse == "" {
			return nil, false
		}
		step.Kind = StepDequantize
		step.Detail = source.Inverse
	default:
		if source.Inverse == "" {
			return nil, false
		}
		step.Kind = StepRequantize
		step.Detail = source.Inverse + " then " + target.Inverse
	}
	return &step, true
}
