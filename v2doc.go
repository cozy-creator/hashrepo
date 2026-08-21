package tensorfs

// Tensor-layout contract v2 — the shared document substrate.
//
// v1 shipped ONE kind of document (a shapeless pattern list) and three engines
// that read it. v2 ships THREE kinds of document and one engine:
//
//	TOPOLOGY   a finite map {key -> logical shape}, dtype-free, shard-flattened,
//	           derived MECHANICALLY from a reference checkpoint's headers.
//	QUANT RULE a function on topologies: an eligibility predicate computable
//	           from shapes, per-tensor emissions whose shapes are pure functions
//	           of the input dims, and the dequant equation that makes
//	           derivability computable.
//	MORPHISM   a topology -> topology map: rekey tables and fusion seams. With
//	           quant rules, the ONLY hand-authored artifacts.
//
// LAYOUT = quant(topology) is COMPUTED, never stored. That is the whole
// compression: N topologies + M rules replace the N x M documents v1 needed.
//
// This file carries what all three share: the handle grammar, the digest, and
// the ONE evaluator for emission-shape formulas. There is exactly one because
// a second would be a second answer to "what header should these bytes have".

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Document format tags. A document says what it is in its first field; a
// reader that guesses from shape would accept a topology as a rule.
const (
	TopologyFormat  = "tensorfs-topology-v2"
	QuantRuleFormat = "tensorfs-quant-rule-v2"
	MorphismFormat  = "tensorfs-morphism-v2"
)

// LayoutError is a typed refusal with a stable kebab-case reason, the same
// shape v1's ContractError had — operators and tests switch on the reason.
type LayoutError struct {
	Reason  string
	Message string
}

func (e *LayoutError) Error() string { return "layout: " + e.Reason + ": " + e.Message }

func refuse(reason, format string, arguments ...any) error {
	return &LayoutError{Reason: reason, Message: fmt.Sprintf(format, arguments...)}
}

// --- handles ---------------------------------------------------------------

// Handle is a document's identity: `<name>@<version>`.
//
// The digest is what a stamp is PINNED by; the handle is how a human and a
// lane declaration spell it. Both travel together — a handle whose document
// digest changed is a different document wearing a familiar name, which is the
// failure v1's four silent digest changes were forgiven for only because
// nothing had stamped them yet.
type Handle struct {
	Name    string
	Version uint32
}

func (h Handle) String() string { return fmt.Sprintf("%s@%d", h.Name, h.Version) }

// ParseHandle reads `<name>@<version>`. The name grammar is TFM1's, because a
// manifest has to be able to carry it.
func ParseHandle(text string) (Handle, error) {
	name, digits, found := strings.Cut(text, "@")
	if !found {
		return Handle{}, refuse("handle", "%q has no @version", text)
	}
	version, err := strconv.ParseUint(digits, 10, 32)
	if err != nil || version == 0 {
		return Handle{}, refuse("handle", "%q has no usable version", text)
	}
	if name == "" || len(name) > TFM1MaxContractNameBytes || !isTFM1ContractName(name) {
		return Handle{}, refuse("handle", "%q is not a usable name", text)
	}
	return Handle{Name: name, Version: uint32(version)}, nil
}

// LayoutID is a stamp: the (topology, quant) PAIR that identifies a
// checkpoint's bytes. th#1809 settled the wire rendering as `<topology>+<quant>`
// and it is shared with the SDK's render() — a spelling drift here is a
// CAS-address fork, so the join character is not negotiable.
type LayoutID struct {
	Topology Handle
	Quant    Handle
}

func (l LayoutID) String() string { return l.Topology.String() + "+" + l.Quant.String() }

// ParseLayoutID reads the wire rendering back.
func ParseLayoutID(text string) (LayoutID, error) {
	left, right, found := strings.Cut(text, "+")
	if !found {
		return LayoutID{}, refuse("layout-id", "%q is not <topology>+<quant>", text)
	}
	topology, err := ParseHandle(left)
	if err != nil {
		return LayoutID{}, err
	}
	quant, err := ParseHandle(right)
	if err != nil {
		return LayoutID{}, err
	}
	return LayoutID{Topology: topology, Quant: quant}, nil
}

// Equal is FIELD-WISE, never string-wise. Two halves that happen to render the
// same are the same stamp; a renderer bug must not be able to invent a match.
func (l LayoutID) Equal(other LayoutID) bool {
	return l.Topology == other.Topology && l.Quant == other.Quant
}

// digestOf is every v2 document's identity: SHA-256 over a canonical rendering
// the document itself produces. JSON is NOT the digest input — key order,
// spacing and number spelling are not part of what a document MEANS.
func digestOf(canonical string) string {
	sum := sha256.Sum256([]byte(canonical))
	return hex.EncodeToString(sum[:])
}

// --- shapes ----------------------------------------------------------------

// Shape is a LOGICAL row-major shape, outermost axis first. Same convention as
// the safetensors header and the Rust inventory; GGUF `ne` is reversed on read.
type Shape []uint64

func (s Shape) String() string {
	parts := make([]string, 0, len(s))
	for _, dimension := range s {
		parts = append(parts, strconv.FormatUint(dimension, 10))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Equal compares rank then dims. THE THING v1 COULD NOT DO: v1 contracts were
// deliberately shapeless, so SDXL-inpainting's 9-channel conv_in was
// indistinguishable from SDXL's 4-channel one and admitted.
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for at := range s {
		if s[at] != other[at] {
			return false
		}
	}
	return true
}

func (s Shape) clone() Shape {
	out := make(Shape, len(s))
	copy(out, s)
	return out
}

// --- the ONE emission-shape evaluator --------------------------------------
//
// A quant rule's emissions declare shapes as pure functions of the source
// tensor's dims: `d0`, `d1/2` (nvfp4 packs two elements per byte), `d1/16` (a
// per-16-block scale), `ceil(d0/128)*128*ceil(d1/16/4)*4` (a pre-swizzled
// blocked scale array). This is the whole language — integers, `dN`, `*`, `/`
// and `ceil(...)` — and this is its only evaluator, in the whole system.
//
// Division is EXACT outside ceil(). `d1/16` on a tensor whose inner dim is not
// a multiple of 16 is not "rounded"; it means the rule's eligibility predicate
// let through a tensor it should not have, so it refuses.

type shapeExpr struct {
	text string
	// parsed as a flat product/quotient chain; the language is small enough
	// that a tree buys nothing.
	terms []exprTerm
}

type exprTerm struct {
	divide bool
	// exactly one of the three is set
	literal uint64
	axis    int // -1 when not an axis
	inner   *shapeExpr
	ceiling bool
}

func parseShapeExpr(text string) (*shapeExpr, error) {
	expression := &shapeExpr{text: text}
	rest := strings.ReplaceAll(text, " ", "")
	if rest == "" {
		return nil, refuse("shape-formula", "empty formula")
	}
	divide := false
	for len(rest) > 0 {
		var token string
		if strings.HasPrefix(rest, "ceil(") {
			depth, at := 0, 0
			for ; at < len(rest); at++ {
				if rest[at] == '(' {
					depth++
				} else if rest[at] == ')' {
					depth--
					if depth == 0 {
						break
					}
				}
			}
			if depth != 0 {
				return nil, refuse("shape-formula", "%q has an unclosed ceil(", text)
			}
			inner, err := parseShapeExpr(rest[len("ceil("):at])
			if err != nil {
				return nil, err
			}
			expression.terms = append(expression.terms,
				exprTerm{divide: divide, axis: -1, inner: inner, ceiling: true})
			rest = rest[at+1:]
		} else {
			end := strings.IndexAny(rest, "*/")
			if end < 0 {
				token, rest = rest, ""
			} else {
				token, rest = rest[:end], rest[end:]
			}
			term, err := parseExprAtom(token, divide)
			if err != nil {
				return nil, err
			}
			expression.terms = append(expression.terms, term)
		}
		if len(rest) == 0 {
			break
		}
		switch rest[0] {
		case '*':
			divide = false
		case '/':
			divide = true
		default:
			return nil, refuse("shape-formula", "%q has a stray %q", text, rest[:1])
		}
		rest = rest[1:]
		if rest == "" {
			return nil, refuse("shape-formula", "%q ends on an operator", text)
		}
	}
	return expression, nil
}

func parseExprAtom(token string, divide bool) (exprTerm, error) {
	if token == "" {
		return exprTerm{}, refuse("shape-formula", "empty term")
	}
	if token[0] == 'd' {
		axis, err := strconv.Atoi(token[1:])
		if err != nil || axis < 0 {
			return exprTerm{}, refuse("shape-formula", "%q is not an axis", token)
		}
		return exprTerm{divide: divide, axis: axis}, nil
	}
	literal, err := strconv.ParseUint(token, 10, 64)
	if err != nil || literal == 0 {
		return exprTerm{}, refuse("shape-formula", "%q is not a positive integer", token)
	}
	return exprTerm{divide: divide, axis: -1, literal: literal}, nil
}

// eval computes one emitted dimension from the source shape.
func (e *shapeExpr) eval(source Shape) (uint64, error) {
	value := uint64(1)
	for _, term := range e.terms {
		operand := uint64(0)
		switch {
		case term.inner != nil:
			inner, err := term.inner.evalCeiling(source)
			if err != nil {
				return 0, err
			}
			operand = inner
		case term.axis >= 0:
			if term.axis >= len(source) {
				return 0, refuse("shape-formula",
					"%q reads axis %d of a rank-%d tensor", e.text, term.axis, len(source))
			}
			operand = source[term.axis]
		default:
			operand = term.literal
		}
		if term.divide {
			if operand == 0 || value%operand != 0 {
				return 0, refuse("shape-formula",
					"%q divides %d by %d, which is not exact — the eligibility "+
						"predicate admitted a tensor the emission cannot shape",
					e.text, value, operand)
			}
			value /= operand
		} else {
			value *= operand
		}
	}
	return value, nil
}

// evalCeiling is eval with the final division rounded UP — the only place
// inexact division is legal, and it has to be asked for by name.
func (e *shapeExpr) evalCeiling(source Shape) (uint64, error) {
	value := uint64(1)
	for _, term := range e.terms {
		operand := uint64(0)
		switch {
		case term.inner != nil:
			inner, err := term.inner.evalCeiling(source)
			if err != nil {
				return 0, err
			}
			operand = inner
		case term.axis >= 0:
			if term.axis >= len(source) {
				return 0, refuse("shape-formula",
					"%q reads axis %d of a rank-%d tensor", e.text, term.axis, len(source))
			}
			operand = source[term.axis]
		default:
			operand = term.literal
		}
		if term.divide {
			if operand == 0 {
				return 0, refuse("shape-formula", "%q divides by zero", e.text)
			}
			value = (value + operand - 1) / operand
		} else {
			value *= operand
		}
	}
	return value, nil
}

func evalShape(formula []*shapeExpr, source Shape) (Shape, error) {
	out := make(Shape, 0, len(formula))
	for _, expression := range formula {
		value, err := expression.eval(source)
		if err != nil {
			return nil, err
		}
		out = append(out, value)
	}
	return out, nil
}

func sortedMapKeys[V any](in map[string]V) []string {
	keys := make([]string, 0, len(in))
	for key := range in {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
