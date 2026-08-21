package tensorfs

// The v2 CATALOG: every topology record, quant rule and morphism seam file,
// embedded in the binary and parsed once.
//
// Embedding is not a convenience. A bind decision that had to read a directory
// could be given a different directory, and a hub and a worker holding two
// corpora is the drift v1's three matchers already proved is undetectable
// until a pod 500s. The corpus ships WITH the engine or not at all.
//
// The loader VALIDATES the corpus against itself: every morphism is applied to
// its `from` topology and the result must equal the catalog's own `to` record,
// key for key and shape for shape. A seam file whose shares are wrong is
// therefore a red at load rather than a wrong admit at serve.

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
	"sync"
)

//go:embed spec/v2/topologies/*.json spec/v2/rules/*.json spec/v2/morphisms/*.json
//go:embed spec/v2/display-names.json
var v2corpus embed.FS

// Catalog is the parsed corpus plus the derived index the engine searches.
type Catalog struct {
	topologies map[Handle]*Topology
	rules      map[Handle]*QuantRule
	morphisms  map[Handle]*Morphism
	// outgoing maps a topology handle to the morphisms that leave it — the
	// adjacency the derivability search walks.
	outgoing map[Handle][]*Morphism

	// display holds the v1 lane-handle spelling each pair was known by. It is
	// for humans, and it is NEVER PARSED: identity is the digest pair, a
	// declaration names the pair explicitly, and a display name gates nothing.
	// The design keeps it because an operator reading a refusal should still
	// recognise the lane they declared last week.
	display map[LayoutID]string

	layouts     map[LayoutID]*Layout
	layoutsOnce sync.Mutex
	order       []LayoutID // deterministic search order
}

// DisplayName is the pair's human spelling, or its wire rendering when it has
// no older name.
func (c *Catalog) DisplayName(id LayoutID) string {
	if name, found := c.display[id]; found {
		return name
	}
	return id.String()
}

var (
	builtinOnce sync.Once
	builtin     *Catalog
	builtinErr  error
)

// BuiltinCatalog is the embedded corpus, parsed and self-validated once.
func BuiltinCatalog() (*Catalog, error) {
	builtinOnce.Do(func() { builtin, builtinErr = LoadCatalog(v2corpus, "spec/v2") })
	return builtin, builtinErr
}

// LoadCatalog reads a corpus from any filesystem — the embedded one, or a
// directory when a tool is regenerating it.
func LoadCatalog(files fs.FS, root string) (*Catalog, error) {
	catalog := &Catalog{
		topologies: map[Handle]*Topology{}, rules: map[Handle]*QuantRule{},
		morphisms: map[Handle]*Morphism{}, outgoing: map[Handle][]*Morphism{},
		display: map[LayoutID]string{}, layouts: map[LayoutID]*Layout{},
	}
	read := func(directory string, each func(name string, document []byte) error) error {
		entries, err := fs.ReadDir(files, path.Join(root, directory))
		if err != nil {
			return refuse("catalog", "%s: %v", directory, err)
		}
		names := make([]string, 0, len(entries))
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
				names = append(names, entry.Name())
			}
		}
		sort.Strings(names)
		for _, name := range names {
			document, err := fs.ReadFile(files, path.Join(root, directory, name))
			if err != nil {
				return refuse("catalog", "%s/%s: %v", directory, name, err)
			}
			if err := each(name, document); err != nil {
				return err
			}
		}
		return nil
	}

	if err := read("topologies", func(name string, document []byte) error {
		topology, err := ParseTopology(document)
		if err != nil {
			return refuse("catalog", "topologies/%s: %v", name, err)
		}
		if _, duplicate := catalog.topologies[topology.handle]; duplicate {
			return refuse("catalog", "%s is declared twice", topology.handle)
		}
		catalog.topologies[topology.handle] = topology
		return nil
	}); err != nil {
		return nil, err
	}
	if err := read("rules", func(name string, document []byte) error {
		rule, err := ParseQuantRule(document)
		if err != nil {
			return refuse("catalog", "rules/%s: %v", name, err)
		}
		if _, duplicate := catalog.rules[rule.handle]; duplicate {
			return refuse("catalog", "%s is declared twice", rule.handle)
		}
		catalog.rules[rule.handle] = rule
		return nil
	}); err != nil {
		return nil, err
	}
	if err := read("morphisms", func(name string, document []byte) error {
		morphism, err := ParseMorphism(document)
		if err != nil {
			return refuse("catalog", "morphisms/%s: %v", name, err)
		}
		if _, duplicate := catalog.morphisms[morphism.handle]; duplicate {
			return refuse("catalog", "%s is declared twice", morphism.handle)
		}
		catalog.morphisms[morphism.handle] = morphism
		catalog.outgoing[morphism.from] = append(catalog.outgoing[morphism.from], morphism)
		return nil
	}); err != nil {
		return nil, err
	}
	if document, err := fs.ReadFile(files, path.Join(root, "display-names.json")); err == nil {
		var raw struct {
			Names map[string]string `json:"names"`
		}
		if err := json.Unmarshal(document, &raw); err != nil {
			return nil, refuse("catalog", "display-names.json: %v", err)
		}
		for spelling, name := range raw.Names {
			id, err := ParseLayoutID(spelling)
			if err != nil {
				return nil, refuse("catalog", "display-names.json: %v", err)
			}
			catalog.display[id] = name
		}
	}
	if len(catalog.topologies) == 0 || len(catalog.rules) == 0 {
		return nil, refuse("catalog", "a corpus with no topology or no rule decides nothing")
	}
	if err := catalog.validate(); err != nil {
		return nil, err
	}
	catalog.index()
	return catalog, nil
}

// validate closes the loop on the hand-authored half.
func (c *Catalog) validate() error {
	for _, handle := range sortedHandles(c.morphisms) {
		morphism := c.morphisms[handle]
		source, known := c.topologies[morphism.from]
		if !known {
			return refuse("catalog", "%s maps %s, which the corpus does not carry",
				handle, morphism.from)
		}
		computed, err := morphism.Apply(source)
		if err != nil {
			return refuse("catalog", "%s: %v", handle, err)
		}
		target, known := c.topologies[morphism.to]
		if !known {
			return refuse("catalog", "%s produces %s, which the corpus does not carry",
				handle, morphism.to)
		}
		if err := sameTopology(computed, target); err != nil {
			return refuse("catalog",
				"%s does not produce %s: %v. A seam file that is subtly wrong is "+
					"silent at serve — every name and shape correct and every number "+
					"wrong — so it has to be loud here", handle, morphism.to, err)
		}
		if morphism.invertible {
			// An invertible morphism gets its reverse edge for free, and it is
			// PROVED here rather than asserted: running the inverse over the
			// target must land back on the source topology, key for key.
			inverse := morphism.Inverse()
			back, err := inverse.Apply(target)
			if err != nil {
				return refuse("catalog", "%s inverted: %v", handle, err)
			}
			back.handle = source.handle
			if err := sameTopology(back, source); err != nil {
				return refuse("catalog",
					"%s claims to be invertible and is not: %v", handle, err)
			}
			c.outgoing[inverse.from] = append(c.outgoing[inverse.from], inverse)
		}
	}
	return nil
}

func sameTopology(computed, declared *Topology) error {
	if len(computed.components) != len(declared.components) {
		return fmt.Errorf("%d components vs %d",
			len(computed.components), len(declared.components))
	}
	for at := range computed.components {
		left, right := &computed.components[at], &declared.components[at]
		if left.Name != right.Name {
			return fmt.Errorf("component %q vs %q", left.Name, right.Name)
		}
		for _, key := range left.keys {
			shape, found := right.tensors[key]
			if !found {
				return fmt.Errorf("computed %q is not in %s", key, declared.handle)
			}
			if !shape.Equal(left.tensors[key]) {
				return fmt.Errorf("%q computes to %s but %s declares %s",
					key, left.tensors[key], declared.handle, shape)
			}
		}
		for _, key := range right.keys {
			if _, found := left.tensors[key]; !found {
				return fmt.Errorf("%s declares %q, which is not computed", declared.handle, key)
			}
		}
	}
	return nil
}

// index fixes the search order. Stamping walks (topology, rule) pairs and the
// FIRST exact match wins, so the order has to be stable across runs or a
// refusal would name a different tensor each time.
func (c *Catalog) index() {
	for _, topology := range sortedHandles(c.topologies) {
		for _, rule := range sortedHandles(c.rules) {
			c.order = append(c.order, LayoutID{Topology: topology, Quant: rule})
		}
	}
}

func sortedHandles[V any](in map[Handle]V) []Handle {
	out := make([]Handle, 0, len(in))
	for handle := range in {
		out = append(out, handle)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].Version < out[j].Version
	})
	return out
}

// --- lookups ---------------------------------------------------------------

// Topology, Rule and Morphism look one up by handle.
func (c *Catalog) Topology(handle Handle) *Topology { return c.topologies[handle] }
func (c *Catalog) Rule(handle Handle) *QuantRule    { return c.rules[handle] }
func (c *Catalog) Morphism(handle Handle) *Morphism { return c.morphisms[handle] }

// Topologies, Rules, Morphisms enumerate the corpus in canonical order.
func (c *Catalog) Topologies() []Handle { return sortedHandles(c.topologies) }
func (c *Catalog) Rules() []Handle      { return sortedHandles(c.rules) }
func (c *Catalog) Morphisms() []Handle  { return sortedHandles(c.morphisms) }

// Layout computes quant(topology), memoized. The memo is the only cache in the
// engine and it holds a pure function of two documents, so it can never be
// stale in a way a restart would fix.
func (c *Catalog) Layout(id LayoutID) (*Layout, error) {
	c.layoutsOnce.Lock()
	defer c.layoutsOnce.Unlock()
	if cached, found := c.layouts[id]; found {
		return cached, nil
	}
	topology, known := c.topologies[id.Topology]
	if !known {
		return nil, refuse("unknown-topology", "%s", id.Topology)
	}
	rule, known := c.rules[id.Quant]
	if !known {
		return nil, refuse("unknown-quant", "%s", id.Quant)
	}
	layout, err := rule.Apply(topology)
	if err != nil {
		return nil, err
	}
	c.layouts[id] = layout
	return layout, nil
}
