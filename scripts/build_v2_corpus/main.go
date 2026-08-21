// Build the v2 TOPOLOGY records — mechanically, from banked reference headers.
//
// Topologies are never hand-authored. This is the tool that authors them, and
// its whole job is to be boring: read one checkpoint's headers, group the
// members into components, flatten the shards, count the dominant element type,
// record what disagrees with it as islands, write the record with its digest.
//
// A second source is allowed and is still not authoring: a topology may be
// DERIVED by applying a ratified morphism to another topology. The native
// MiniMax-H3 packaging is the diffusers one with 56 head-major qkv triples
// fused, and there is no reachable checkpoint to extract it from — but there IS
// a ratified seam file, and applying it is a derivation, not a guess. The
// catalog loader re-applies the morphism at load and refuses if the record and
// the seam disagree.
//
//	go run ./scripts/build_v2_corpus [name ...]
package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type bankedTensor struct {
	Name   string   `json:"name"`
	Dtype  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	Length uint64   `json:"length"`
}

type bankedFile struct {
	Path    string         `json:"path"`
	Tensors []bankedTensor `json:"tensors"`
}

type bankedHeaders struct {
	ID     string       `json:"id"`
	Source string       `json:"source"`
	Note   string       `json:"note"`
	Files  []bankedFile `json:"files"`
}

type entry struct {
	name, version, source, description string
}

func main() {
	root := "."
	wanted := map[string]bool{}
	for _, argument := range os.Args[1:] {
		wanted[argument] = true
	}
	entries := readCorpus(filepath.Join(root, "spec", "v2", "CORPUS.tsv"))
	// Derived topologies need their source already written, so headers first.
	for _, pass := range []string{"headers:", "morphism:", "morphism-inverse:"} {
		for _, item := range entries {
			if !strings.HasPrefix(item.source, pass) {
				continue
			}
			if len(wanted) > 0 && !wanted[item.name] {
				continue
			}
			build(root, item)
		}
	}
	// The hand-authored documents get their digests PINNED INTO THEM. A rule or
	// a seam file with no digest accepts any edit silently, and these two are
	// the only documents a human types — which is exactly where a silent edit
	// would do the most damage.
	pinDigests(root)
	// The catalog re-reads what was written and re-applies every morphism; a
	// build that "succeeded" into an inconsistent corpus is not a build.
	catalog, err := tensorfs.LoadCatalog(os.DirFS(root), "spec/v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "CORPUS REFUSED: %v\n", err)
		os.Exit(1)
	}
	writeDigestVectors(root, catalog)
	fmt.Fprintf(os.Stderr, "corpus: %d topologies, %d rules, %d morphisms\n",
		len(catalog.Topologies()), len(catalog.Rules()), len(catalog.Morphisms()))
}

// writeDigestVectors is v2's conformance corpus, and it is one file because
// there is far less to conform to: v1 needed cross-language vectors because
// three engines parsed its documents and had to agree byte for byte on a
// canonical rendering. v2 has one engine. What any reader still needs to check
// is that the documents it holds are the documents that were pinned, so the
// vectors are the digests.
func writeDigestVectors(root string, catalog *tensorfs.Catalog) {
	document := map[string]any{
		"_": "Every catalogued document's digest. A reader that holds a corpus " +
			"checks it against this; a change here is a change to what a stamp " +
			"MEANS and must be a version bump, not an edit.",
		"topologies": map[string]string{},
		"rules":      map[string]string{},
		"morphisms":  map[string]string{},
	}
	for _, handle := range catalog.Topologies() {
		document["topologies"].(map[string]string)[handle.String()] =
			catalog.Topology(handle).Digest()
	}
	for _, handle := range catalog.Rules() {
		document["rules"].(map[string]string)[handle.String()] = catalog.Rule(handle).Digest()
	}
	for _, handle := range catalog.Morphisms() {
		document["morphisms"].(map[string]string)[handle.String()] =
			catalog.Morphism(handle).Digest()
	}
	out, err := json.MarshalIndent(document, "", " ")
	if err != nil {
		panic(err)
	}
	target := filepath.Join(root, "spec", "v2", "vectors", "digests.json")
	if err := os.WriteFile(target, append(out, '\n'), 0o644); err != nil {
		panic(err)
	}
}

func pinDigests(root string) {
	for _, directory := range []string{"rules", "morphisms"} {
		entries, err := filepath.Glob(filepath.Join(root, "spec", "v2", directory, "*.json"))
		if err != nil {
			panic(err)
		}
		for _, path := range entries {
			raw, err := os.ReadFile(path)
			if err != nil {
				panic(err)
			}
			// Re-pin from the document WITHOUT its stored digest: a rendering
			// change (a new field in the canonical form) must be able to
			// re-pin, and a parse that enforced the stale pin could never.
			var document map[string]any
			if err := json.Unmarshal(raw, &document); err != nil {
				panic(err)
			}
			delete(document, "digest")
			unpinned, err := json.Marshal(document)
			if err != nil {
				panic(err)
			}
			var digest string
			if directory == "rules" {
				rule, err := tensorfs.ParseQuantRule(unpinned)
				if err != nil {
					panic(fmt.Sprintf("%s: %v", path, err))
				}
				digest = rule.Digest()
			} else {
				morphism, err := tensorfs.ParseMorphism(unpinned)
				if err != nil {
					panic(fmt.Sprintf("%s: %v", path, err))
				}
				digest = morphism.Digest()
			}
			var stored map[string]any
			if err := json.Unmarshal(raw, &stored); err != nil {
				panic(err)
			}
			document = stored
			if existing, found := document["digest"]; found && existing == digest {
				continue
			}
			document["digest"] = digest
			out, err := json.MarshalIndent(document, "", " ")
			if err != nil {
				panic(err)
			}
			if err := os.WriteFile(path, append(out, '\n'), 0o644); err != nil {
				panic(err)
			}
			fmt.Fprintf(os.Stderr, "[pin] %s %s\n", filepath.Base(path), digest[:16])
		}
	}
}

func build(root string, item entry) {
	version, err := strconv.ParseUint(item.version, 10, 32)
	if err != nil {
		panic(fmt.Sprintf("%s: bad version %q", item.name, item.version))
	}
	var topology *tensorfs.Topology
	switch {
	case strings.HasPrefix(item.source, "headers:"):
		identifier := strings.TrimPrefix(item.source, "headers:")
		headers := readHeaders(filepath.Join(
			root, "spec", "v2", "headers", identifier+".json.gz"))
		files := make([]tensorfs.ArtifactFile, 0, len(headers.Files))
		for _, member := range headers.Files {
			tensors := make([]tensorfs.InventoryTensor, 0, len(member.Tensors))
			for _, tensor := range member.Tensors {
				tensors = append(tensors, tensorfs.InventoryTensor{
					Name: tensor.Name, Dtype: tensor.Dtype,
					Shape: tensor.Shape, Length: tensor.Length,
				})
			}
			files = append(files, tensorfs.ArtifactFile{Path: member.Path, Tensors: tensors})
		}
		topology, err = tensorfs.TopologyFromHeaders(item.name, uint32(version),
			identifier+" ("+headers.Source+")", files)
		if err != nil {
			panic(fmt.Sprintf("%s: %v", item.name, err))
		}
	case strings.HasPrefix(item.source, "morphism:"),
		strings.HasPrefix(item.source, "morphism-inverse:"):
		backwards := strings.HasPrefix(item.source, "morphism-inverse:")
		spelled := strings.TrimPrefix(strings.TrimPrefix(item.source, "morphism-inverse:"), "morphism:")
		handle, parseErr := tensorfs.ParseHandle(spelled)
		if parseErr != nil {
			panic(parseErr)
		}
		document, readErr := os.ReadFile(morphismPath(root, handle))
		if readErr != nil {
			panic(readErr)
		}
		morphism, parseErr := tensorfs.ParseMorphism(document)
		if parseErr != nil {
			panic(parseErr)
		}
		if backwards {
			morphism = morphism.Inverse()
		}
		sourceDocument, readErr := os.ReadFile(topologyPath(root, morphism.From()))
		if readErr != nil {
			panic(fmt.Sprintf("%s needs %s first: %v", item.name, morphism.From(), readErr))
		}
		source, parseErr := tensorfs.ParseTopology(sourceDocument)
		if parseErr != nil {
			panic(parseErr)
		}
		topology, err = morphism.Apply(source)
		if err != nil {
			panic(fmt.Sprintf("%s: %v", item.name, err))
		}
	default:
		panic(fmt.Sprintf("%s: unrecognised source %q", item.name, item.source))
	}
	rendered, err := renderWith(topology, item.description)
	if err != nil {
		panic(err)
	}
	target := topologyPath(root, topology.Handle())
	if err := os.WriteFile(target, rendered, 0o644); err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "[%s] %d component(s), %d tensors -> %s\n",
		topology.Handle(), len(topology.Components()), topology.Tensors(),
		filepath.Base(target))
}

// renderWith re-renders a topology carrying the corpus's description. The
// description is the ONE human field on a mechanically derived record, and it
// is outside the digest for exactly that reason.
func renderWith(topology *tensorfs.Topology, description string) ([]byte, error) {
	rendered, err := topology.Render()
	if err != nil {
		return nil, err
	}
	var document map[string]any
	if err := json.Unmarshal(rendered, &document); err != nil {
		return nil, err
	}
	document["description"] = description
	out, err := json.MarshalIndent(document, "", " ")
	if err != nil {
		return nil, err
	}
	return append(out, '\n'), nil
}

func topologyPath(root string, handle tensorfs.Handle) string {
	return filepath.Join(root, "spec", "v2", "topologies",
		fmt.Sprintf("%s.v%d.json", handle.Name, handle.Version))
}

func morphismPath(root string, handle tensorfs.Handle) string {
	return filepath.Join(root, "spec", "v2", "morphisms",
		fmt.Sprintf("%s.v%d.json", handle.Name, handle.Version))
}

func readCorpus(path string) []entry {
	raw, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	var out []entry
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) < 4 {
			panic(fmt.Sprintf("CORPUS.tsv: %q wants name/version/source/description", line))
		}
		out = append(out, entry{
			name: fields[0], version: fields[1], source: fields[2],
			description: strings.Join(fields[3:], " "),
		})
	}
	return out
}

func readHeaders(path string) bankedHeaders {
	handle, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer func() { _ = handle.Close() }()
	reader, err := gzip.NewReader(handle)
	if err != nil {
		panic(err)
	}
	var headers bankedHeaders
	if err := json.NewDecoder(reader).Decode(&headers); err != nil {
		panic(fmt.Sprintf("%s: %v", path, err))
	}
	return headers
}
