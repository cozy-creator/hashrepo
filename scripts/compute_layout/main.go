// Print quant(topology) — the computed expected header — as JSON.
//
// This is the ONE evaluator, made callable from outside Go. The hub links the
// package and calls `Catalog.Layout` directly; a worker, a producer or a test
// that is not in Go reads what this prints. Neither grows an evaluator of its
// own, which is the whole point of the design's "the hub calls it and never
// grows a second one".
//
//	go run ./scripts/compute_layout 'sdxl.diffusers@1+cozy.fp8-rowwise@1'
//	go run ./scripts/compute_layout --list
package main

import (
	"encoding/json"
	"fmt"
	"os"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type outTensor struct {
	Dtypes   []string `json:"dtypes"`
	Shape    []uint64 `json:"shape"`
	Optional bool     `json:"optional,omitempty"`
}

type outComponent struct {
	Name    string               `json:"name"`
	Role    string               `json:"role"`
	Tensors map[string]outTensor `json:"tensors"`
}

type outQuant struct {
	Handle string `json:"handle"`
	// DeclaredDtype and CapabilityFloorSM are the IDENTITY FACTS a worker
	// reads. Under v1 they were a document field plus a lookup table keyed on
	// the spelling, and a lane could lose its floor by being spelled
	// differently; here they are properties of the rule itself.
	DeclaredDtype     string            `json:"declared_dtype"`
	CapabilityFloorSM int               `json:"capability_floor_sm"`
	Conventions       map[string]string `json:"conventions"`
	Lossy             bool              `json:"lossy"`
	Inverse           string            `json:"inverse,omitempty"`
	Transformed       int               `json:"transformed"`
	Digest            string            `json:"digest"`
}

type outLayout struct {
	Stamp    string `json:"stamp"`
	Topology string `json:"topology"`
	// TopologyDigest pins the record the layout was computed from. Two hubs
	// that computed a layout from two different records must not be able to
	// call the result the same thing.
	TopologyDigest string         `json:"topology_digest"`
	Quant          outQuant       `json:"quant"`
	Components     []outComponent `json:"components"`
}

func main() {
	catalog, err := tensorfs.BuiltinCatalog()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(os.Args) == 2 && os.Args[1] == "--list" {
		for _, topology := range catalog.Topologies() {
			for _, rule := range catalog.Rules() {
				id := tensorfs.LayoutID{Topology: topology, Quant: rule}
				layout, err := catalog.Layout(id)
				if err != nil {
					continue
				}
				if catalog.Rule(rule).Transforms() && layout.Transformed() == 0 {
					// Not a layout: the rule finds nothing eligible here, so
					// this pair computes its base plain rule's header.
					continue
				}
				fmt.Printf("%-64s %6d tensors\n", id, layout.Tensors())
			}
		}
		return
	}
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: compute_layout '<topology>@v+<quant>@v' | --list")
		os.Exit(2)
	}
	id, err := tensorfs.ParseLayoutID(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	layout, err := catalog.Layout(id)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	rule := catalog.Rule(id.Quant)
	document := outLayout{
		Stamp: id.String(), Topology: id.Topology.String(),
		TopologyDigest: catalog.Topology(id.Topology).Digest(),
		Quant: outQuant{
			Handle: id.Quant.String(), DeclaredDtype: rule.DeclaredDtype,
			CapabilityFloorSM: rule.CapabilityFloorSM, Conventions: rule.Conventions,
			Lossy: rule.Lossy, Inverse: rule.Inverse,
			Transformed: layout.Transformed(), Digest: rule.Digest(),
		},
	}
	for at := range layout.Components {
		component := &layout.Components[at]
		mapped := outComponent{
			Name: component.Name, Role: string(component.Role),
			Tensors: make(map[string]outTensor, len(component.Tensors)),
		}
		for key, entry := range component.Tensors {
			mapped.Tensors[key] = outTensor{
				Dtypes: entry.Dtypes, Shape: entry.Shape, Optional: entry.Optional,
			}
		}
		document.Components = append(document.Components, mapped)
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", " ")
	if err := encoder.Encode(document); err != nil {
		panic(err)
	}
}
