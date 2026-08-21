package tensorfs_test

// PROOF BATTERY ITEM 1 — THE REGRESSION.
//
// "v2 admission reproduces every verdict the v1 engine gives over the existing
// corpus + real headers. Intentional diffs are enumerated in the fixture, not
// waved through." (tensor-layout-v2 design §6.1)
//
// The baseline is spec/v2/baselines/v1-verdicts.json: 600 answers from the real
// `Contract.Verdict` call tensorhub's bind gate makes, over all 30 v1 documents
// x all 20 banked checkpoints, banked BEFORE v1 was deleted. The cross product
// is the point — a baseline of "each document against its own reference" only
// ever exercises `satisfies`, and an engine that answered `satisfies` to
// everything would reproduce it perfectly.
//
// v1's three answers map onto v2's three:
//
//	satisfies    -> admit
//	derivable    -> derivable
//	incompatible -> refuse
//
// Every departure is listed in spec/v2/baselines/successors.json with the
// reason it is CORRECT, and this test fails on an unlisted one AND on a listed
// one that no longer fires. A fixture that outlives its diff is a fixture that
// asserts nothing.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type v1Baseline struct {
	Contract   string `json:"contract"`
	Checkpoint string `json:"checkpoint"`
	Kind       string `json:"kind"`
	Matched    int    `json:"matched"`
	Rendered   string `json:"rendered"`
}

type successorMap struct {
	Lanes        map[string]string `json:"lanes"`
	NoSuccessor  map[string]string `json:"no_successor"`
	Intentional  []intentionalDiff `json:"intentional_diffs"`
	NoteFragment string            `json:"fragments_note"`
}

type intentionalDiff struct {
	Contract   string `json:"contract"`
	Checkpoint string `json:"checkpoint"`
	V1         string `json:"v1"`
	V2         string `json:"v2"`
	Why        string `json:"why"`
}

func loadSuccessors(t *testing.T) successorMap {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "baselines", "successors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var out successorMap
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func loadV1Baselines(t *testing.T) []v1Baseline {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "baselines", "v1-verdicts.json"))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Contracts   int          `json:"contracts"`
		Checkpoints int          `json:"checkpoints"`
		Verdicts    []v1Baseline `json:"verdicts"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		t.Fatal(err)
	}
	if len(document.Verdicts) != document.Contracts*document.Checkpoints {
		t.Fatalf("the bank holds %d verdicts for %d x %d — it is not the full cross "+
			"product any more", len(document.Verdicts), document.Contracts, document.Checkpoints)
	}
	return document.Verdicts
}

func expectedV2(v1 string) tensorfs.DecisionKind {
	switch v1 {
	case "satisfies":
		return tensorfs.DecisionAdmit
	case "derivable":
		return tensorfs.DecisionDerivable
	default:
		return tensorfs.DecisionRefuse
	}
}

func TestV2ReproducesTheBankedV1Verdicts(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	baselines := loadV1Baselines(t)
	successors := loadSuccessors(t)

	fired := map[string]intentionalDiff{}
	listed := map[string]intentionalDiff{}
	for _, diff := range successors.Intentional {
		listed[diff.Contract+"|"+diff.Checkpoint] = diff
	}

	counts := map[string]int{}
	var unlisted []string
	for _, baseline := range baselines {
		lane, hasLane := successors.Lanes[baseline.Contract]
		if !hasLane {
			if _, known := successors.NoSuccessor[baseline.Contract]; !known {
				t.Errorf("%s has neither a v2 lane nor a recorded reason for not "+
					"having one", baseline.Contract)
			}
			counts["no-successor"]++
			continue
		}
		checkpoint, banked := headers[baseline.Checkpoint]
		if !banked {
			t.Fatalf("the bank names checkpoint %q, which is not on disk", baseline.Checkpoint)
		}
		decision := loaded.Admit(checkpoint.artifact(),
			[]tensorfs.LayoutID{mustLayoutID(t, lane)})
		want := expectedV2(baseline.Kind)
		key := baseline.Contract + "|" + baseline.Checkpoint
		if decision.Kind == want {
			counts["reproduced"]++
			if _, isListed := listed[key]; isListed {
				t.Errorf("%s is listed as an intentional diff and did NOT diverge — "+
					"a stale entry makes the fixture assert less than it claims", key)
			}
			continue
		}
		diff, isListed := listed[key]
		if !isListed {
			unlisted = append(unlisted, fmt.Sprintf(
				"  {\"contract\": %q, \"checkpoint\": %q, \"v1\": %q, \"v2\": %q, \"why\": \"\"},\n"+
					"      v1 said: %s\n      v2 says: %s",
				baseline.Contract, baseline.Checkpoint, baseline.Kind, decision.Kind,
				baseline.Rendered, decisionText(decision)))
			continue
		}
		if diff.V1 != baseline.Kind || diff.V2 != string(decision.Kind) {
			t.Errorf("%s is listed as %s -> %s but ran %s -> %s",
				key, diff.V1, diff.V2, baseline.Kind, decision.Kind)
		}
		if diff.Why == "" {
			t.Errorf("%s is listed with no reason", key)
		}
		fired[key] = diff
		counts["intentional"]++
	}
	for key := range listed {
		if _, ran := fired[key]; !ran {
			t.Errorf("%s is listed as an intentional diff but never ran — its "+
				"contract or checkpoint is gone", key)
		}
	}
	if len(unlisted) > 0 {
		sort.Strings(unlisted)
		t.Errorf("%d UNLISTED divergences from the v1 baseline:\n%s",
			len(unlisted), joinLines(unlisted))
	}
	t.Logf("v1 baseline: %d reproduced, %d enumerated intentional diffs, "+
		"%d rows whose document has no v2 lane", counts["reproduced"],
		counts["intentional"], counts["no-successor"])

	// The regression must not be vacuous: if v2 answered `refuse` to
	// everything it would still reproduce 506 of the 600 rows. Assert the
	// admits and the derivables are there too.
	admits, derivables := 0, 0
	for _, baseline := range baselines {
		lane, hasLane := successors.Lanes[baseline.Contract]
		if !hasLane {
			continue
		}
		decision := loaded.Admit(headers[baseline.Checkpoint].artifact(),
			[]tensorfs.LayoutID{mustLayoutID(t, lane)})
		switch decision.Kind {
		case tensorfs.DecisionAdmit:
			admits++
		case tensorfs.DecisionDerivable:
			derivables++
		}
	}
	if admits == 0 || derivables == 0 {
		t.Fatalf("the run produced %d admits and %d derivables — a comparison "+
			"that only ever sees one arm proves nothing", admits, derivables)
	}
	t.Logf("arms exercised: %d admit, %d derivable", admits, derivables)
}

func decisionText(decision tensorfs.Decision) string {
	if decision.Kind == tensorfs.DecisionRefuse {
		return decision.Refusal.String()
	}
	return decision.String()
}

func joinLines(lines []string) string {
	out := ""
	for _, line := range lines {
		out += line + "\n"
	}
	return out
}
