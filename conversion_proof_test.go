package tensorfs_test

// The conversion producer, graded by the matcher the hub actually calls.
//
// The producer is Python (`tensorfs.convert`) because that is where a tensor
// kernel can live; the verdict is Go (`Contract.Verdict`) because that is what
// tensorhub's bind gate consumes. Grading the producer with a Python
// re-implementation of the verdict would be a third matcher marking its own
// homework, and the failure it would hide is the expensive one: a converted
// checkpoint that the producer believes satisfies the lane and the gate refuses.
//
// So this test reads the REAL headers of the trees the producer wrote — emitted
// by `test_conversion.py::test_write_the_verdict_proof_fixture` — and asserts
// the two verdicts th#2164 exists for:
//
//	fp16-packaged SDXL vs the bf16 lane   -> derivable, naming a dtype-cast
//	the SAME tree after conversion        -> satisfies
//
// Driven by `scripts/prove-conversion.sh`, which runs both halves in order.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type proofTensor struct {
	Name   string   `json:"name"`
	Dtype  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	Length uint64   `json:"length"`
}

type proofCase struct {
	Lane      string                   `json:"lane"`
	Before    map[string][]proofTensor `json:"before"`
	After     map[string][]proofTensor `json:"after"`
	Rewritten []string                 `json:"rewritten"`
	Claimed   map[string]int           `json:"python_claimed"`
}

func loadProof(t *testing.T) map[string]proofCase {
	t.Helper()
	dir := os.Getenv("TENSORFS_CONVERSION_PROOF_DIR")
	if dir == "" {
		t.Skip("set TENSORFS_CONVERSION_PROOF_DIR, or run scripts/prove-conversion.sh which sets it")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "verdict-cases.json"))
	if err != nil {
		t.Fatalf("the producer half did not write its fixture: %v", err)
	}
	var cases map[string]proofCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("fixture is not readable: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("fixture is empty — a proof that asserts nothing is worse than no proof")
	}
	return cases
}

func artifact(files map[string][]proofTensor) []tensorfs.ArtifactFile {
	out := make([]tensorfs.ArtifactFile, 0, len(files))
	for path, tensors := range files {
		members := make([]tensorfs.InventoryTensor, 0, len(tensors))
		for _, tensor := range tensors {
			members = append(members, tensorfs.InventoryTensor{
				Name:   tensor.Name,
				Dtype:  tensor.Dtype,
				Shape:  tensor.Shape,
				Length: tensor.Length,
			})
		}
		out = append(out, tensorfs.ArtifactFile{Path: path, Tensors: members})
	}
	return out
}

func laneContract(t *testing.T, stamp string) *tensorfs.Contract {
	t.Helper()
	name := stamp
	for i := 0; i < len(stamp); i++ {
		if stamp[i] == '@' {
			name = stamp[:i]
			break
		}
	}
	document, err := os.ReadFile(filepath.Join("spec", "v1", "contracts", name+".v1.json"))
	if err != nil {
		t.Fatalf("lane %s: %v", stamp, err)
	}
	contract, err := tensorfs.ParseContract(document)
	if err != nil {
		t.Fatalf("lane %s does not parse: %v", stamp, err)
	}
	return contract
}

// The whole point of th#2164: the CONVERTIBLE verdict stops being a dead branch
// only when something can produce the checkpoint that turns it into Satisfies.
func TestTheConversionTurnsDerivableIntoSatisfies(t *testing.T) {
	for label, proof := range loadProof(t) {
		t.Run(label, func(t *testing.T) {
			contract := laneContract(t, proof.Lane)

			before := contract.Verdict(artifact(proof.Before))
			if before.Kind != tensorfs.VerdictDerivable {
				t.Fatalf("source tree: got %s (%s), want derivable — this case only "+
					"tests a conversion if the source needed one", before.Kind, before)
			}
			if before.Conversion == nil {
				t.Fatal("a derivable verdict with no named conversion is exactly the " +
					"dead branch th#2164 recorded: nothing to enqueue")
			}
			if before.Mismatch == nil || before.Mismatch.Tensor == "" || before.Mismatch.Pattern == "" {
				t.Fatalf("the verdict must name the tensor and the pattern: %+v", before.Mismatch)
			}
			// The offered recipe must be the one the PRODUCER will run. An fp8
			// lane reported as a plain cast is the expensive version of this
			// bug: the job runs, the bytes come back fp8 with no per-row
			// scales, and the half-quantized checkpoint serves wrong numbers
			// with every name, dtype and shape correct.
			if before.Conversion.Kind != contract.Recipe() {
				t.Fatalf("offered %q but %s's bytes are made by %q",
					before.Conversion.Kind, proof.Lane, contract.Recipe())
			}
			t.Logf("source:    %s", before)

			after := contract.Verdict(artifact(proof.After))
			if after.Kind != tensorfs.VerdictSatisfies {
				t.Fatalf("converted tree: got %s (%s), want satisfies", after.Kind, after)
			}
			if after.Explained == 0 || after.Matched == 0 {
				t.Fatalf("satisfies with nothing explained is a silent skip, not a match: %+v", after)
			}
			t.Logf("converted: %s (rewritten: %v)", after, proof.Rewritten)
		})
	}
}

// The producer rewrote only what it had to. A conversion that re-admits an
// untouched member is not wrong, it is expensive — and at checkpoint scale the
// difference is the whole reason the conversion lives beside the CAS.
func TestTheConversionRewritesOnlyWhatItMustAndTheRestStillMatches(t *testing.T) {
	for label, proof := range loadProof(t) {
		t.Run(label, func(t *testing.T) {
			rewritten := map[string]bool{}
			for _, path := range proof.Rewritten {
				rewritten[path] = true
			}
			for path, after := range proof.After {
				if rewritten[path] {
					continue
				}
				before, ok := proof.Before[path]
				if !ok {
					t.Fatalf("%s appeared from nowhere", path)
				}
				if len(before) != len(after) {
					t.Fatalf("%s was not rewritten but its tensor count moved: %d -> %d",
						path, len(before), len(after))
				}
				for i := range before {
					if !reflect.DeepEqual(before[i], after[i]) {
						t.Fatalf("%s was not rewritten but %s changed: %+v -> %+v",
							path, before[i].Name, before[i], after[i])
					}
				}
			}
		})
	}
}

// The planner's Python port of the claim rule, re-counted by the Go matcher.
//
// tensorfs ships ONE matching semantics and, today, three implementations of
// it: Rust (`Contract::matches`, the authority), Go (`match.go`, what the hub
// calls), and the Python claim step the conversion planner needs because the
// Rust one is not reachable from Python. Three is one too many, and the
// standing intent is to collapse the third into a pyo3 binding (tensorfs#128).
// Until then this is the fence, and it is not decorative: while this suite was
// being written the Python port read every two-segment pattern (`a.{i}.b` —
// most of the library) as ambiguous and claimed NOTHING, which turned every
// conversion into a no-op that still returned a plan and still looked like a
// pass. A drift here is silent by construction; only a second implementation
// counting the same tensors makes it loud.
func TestThePythonClaimRuleAgreesWithTheGoMatcher(t *testing.T) {
	for label, proof := range loadProof(t) {
		t.Run(label, func(t *testing.T) {
			if len(proof.Claimed) == 0 {
				t.Fatal("no claim counts in the fixture — the fence is not fencing")
			}
			contract := laneContract(t, proof.Lane)
			for _, file := range artifact(proof.After) {
				match, mismatch := contract.Match(file.Tensors)
				if mismatch != nil {
					t.Fatalf("%s: the converted member must match: %s", file.Path, mismatch)
				}
				if want, ok := proof.Claimed[file.Path]; !ok {
					t.Fatalf("%s: no python count", file.Path)
				} else if match.Matched != want {
					t.Fatalf("%s: go explains %d tensors, python claims %d — the two "+
						"implementations of one rule disagree", file.Path, match.Matched, want)
				}
			}
		})
	}
}
