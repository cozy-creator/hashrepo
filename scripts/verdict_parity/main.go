// The GO half of the verdict parity proof: read the shared fixture corpus,
// render each verdict with the Go implementation, print one line per case.
//
// The Python half (scripts/prove-verdict-parity.sh) does the same through the
// pyo3 binding and diffs the two outputs. Comparing RENDERED verdicts rather
// than struct fields is deliberate: the rendering is what an operator reads, so
// a divergence anywhere in kind, stamp, counts, conversion, mismatch or file
// shows up as a diff instead of being one field nobody thought to assert on.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type tensor struct {
	Name   string   `json:"name"`
	Dtype  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	Length uint64   `json:"length"`
}

type file struct {
	Path    string   `json:"path"`
	Tensors []tensor `json:"tensors"`
}

type testCase struct {
	Name     string `json:"name"`
	For      string `json:"for"`
	Contract string `json:"contract"`
	Files    []file `json:"files"`
}

type corpus struct {
	Cases []testCase `json:"cases"`
}

func main() {
	root := os.Args[1]
	raw, err := os.ReadFile(filepath.Join(root, "scripts", "verdict-fixtures.json"))
	if err != nil {
		panic(err)
	}
	var loaded corpus
	if err := json.Unmarshal(raw, &loaded); err != nil {
		panic(err)
	}
	for _, entry := range loaded.Cases {
		document, err := os.ReadFile(filepath.Join(
			root, "spec", "v1", "contracts", stampToFile(entry.Contract)))
		if err != nil {
			panic(err)
		}
		contract, err := tensorfs.ParseContract(document)
		if err != nil {
			panic(err)
		}
		members := make([]tensorfs.ArtifactFile, 0, len(entry.Files))
		for _, member := range entry.Files {
			tensors := make([]tensorfs.InventoryTensor, 0, len(member.Tensors))
			for _, item := range member.Tensors {
				tensors = append(tensors, tensorfs.InventoryTensor{
					Name: item.Name, Dtype: item.Dtype,
					Shape: item.Shape, Length: item.Length,
				})
			}
			members = append(members, tensorfs.ArtifactFile{Path: member.Path, Tensors: tensors})
		}
		verdict := contract.Verdict(members)
		if string(verdict.Kind) != entry.For {
			fmt.Fprintf(os.Stderr,
				"FIXTURE DRIFT: %s is labelled %q but answered %q — the corpus no longer "+
					"exercises the arm it claims to\n", entry.Name, entry.For, verdict.Kind)
			os.Exit(2)
		}
		fmt.Printf("%s\t%s\t%s\n", entry.Name, contract.Recipe(), verdict.String())
	}
}

// stampToFile maps `name@version` to the document that ships it.
func stampToFile(stamp string) string {
	for i := len(stamp) - 1; i >= 0; i-- {
		if stamp[i] == '@' {
			return stamp[:i] + ".v" + stamp[i+1:] + ".json"
		}
	}
	panic("not a name@version stamp: " + stamp)
}
