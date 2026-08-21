package tensorfs

// TOPOLOGY records: the finite {key -> logical shape} map of one packaging.
//
// Three properties, each of which fixes a v1 defect:
//
//   - FINITE and derived MECHANICALLY from a reference checkpoint's headers.
//     v1 documents were hand-authored pattern lists, so a document could
//     describe a checkpoint that does not exist, and the tensorfs#150 lane
//     found three that refused the very checkpoints they were generated from.
//   - SHAPES ARE KEPT. v1 was deliberately shapeless (tensorfs#122 rec 1) and
//     therefore could not tell SDXL-inpainting's 9-channel `conv_in` from
//     SDXL's 4-channel one, nor SD1.5 from SD2 except by rank.
//   - SHARD-FLATTENED. A checkpoint's members are grouped into COMPONENTS by
//     content, so a 3-shard unet and a 1-file unet are one topology. v1
//     matched per member file and needed a `required` flag to survive
//     sharding — the flag that made "required:false everywhere" able to admit
//     anything. Under v2 that class is inexpressible: there is no flag.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
)

// ComponentRole is what a component IS, which is what a quant rule scopes on:
// `cozy.fp8-rowwise` transforms the DENOISER and passes everything else
// through. Roles are derived from the reference tree's own directory names by
// the table below — mechanical, like the rest of extraction.
type ComponentRole string

const (
	RoleDenoiser     ComponentRole = "denoiser"
	RoleVAE          ComponentRole = "vae"
	RoleTextEncoder  ComponentRole = "text_encoder"
	RoleImageEncoder ComponentRole = "image_encoder"
	RoleBackbone     ComponentRole = "backbone"
	RoleOther        ComponentRole = "other"
)

// roleOfDirectory is the extraction-time table. A tree that spells its denoiser
// something new lands in RoleOther and the extractor SAYS SO, so an unknown
// spelling is a visible fact rather than a silent misclassification.
func roleOfDirectory(directory string) ComponentRole {
	// Wan 2.2 ships TWO denoisers as transformer/ and transformer_2/ — a
	// numbered sibling is the same kind of thing as its base, so a trailing
	// _<digits> is stripped before the role is read. Without this the second
	// denoiser is RoleOther, and a quant rule scoped to denoisers transforms
	// half the network.
	base := strings.ToLower(path.Base(directory))
	if at := strings.LastIndex(base, "_"); at > 0 {
		digits := base[at+1:]
		if digits != "" && strings.Trim(digits, "0123456789") == "" {
			base = base[:at]
		}
	}
	switch {
	case base == "unet" || base == "transformer" || base == "denoiser" ||
		base == "diffusion_models" || base == "dit":
		return RoleDenoiser
	case base == "vae" || strings.HasPrefix(base, "vae_"):
		return RoleVAE
	case strings.HasPrefix(base, "text_encoder") || base == "text_encoders":
		return RoleTextEncoder
	case strings.HasPrefix(base, "image_encoder") || base == "vision_tower":
		return RoleImageEncoder
	case base == "." || base == "":
		// A flat checkpoint IS its network: one component, and the thing a
		// serve lane executes.
		return RoleBackbone
	default:
		return RoleOther
	}
}

// TopologyComponent is one component's finite map.
type TopologyComponent struct {
	Name string
	Role ComponentRole
	// Islands are reference keys whose dtype differs from the topology's
	// dominant one: an F32 `logit_scale` on a bf16 tree, an I64 position-id
	// table — or, routinely, a WHOLE COMPONENT the reference ships at another
	// precision (LTX-2.3 carries a bf16 denoiser beside a 1065-tensor fp32 text
	// encoder). A plain rule tolerates an island at EITHER its own dtype or the
	// lane's, because both packagings are shipped and refusing the reference's
	// own would refuse the reference.
	Islands map[string]string

	tensors map[string]Shape
	keys    []string // sorted; the canonical order
}

// Tensors is the finite map, read-only.
func (c *TopologyComponent) Tensors() map[string]Shape { return c.tensors }

// Keys are the component's key names in canonical (sorted) order.
func (c *TopologyComponent) Keys() []string { return c.keys }

// Topology is one named, digest-pinned packaging.
type Topology struct {
	handle      Handle
	description string
	derivedFrom string
	// dtype is the reference checkpoint's DOMINANT element type, across every
	// component. It is not what the topology MEANS — topologies are dtype-free
	// — it is the reference fact a plain rule reads to know what an island is.
	dtype      string
	components []TopologyComponent
	byName     map[string]*TopologyComponent
}

// Dtype is the reference packaging's dominant element type.
func (t *Topology) Dtype() string { return t.dtype }

// Handle is `<name>@<version>`.
func (t *Topology) Handle() Handle { return t.handle }

// Description is free text for humans and refusal messages.
func (t *Topology) Description() string { return t.description }

// DerivedFrom names the reference checkpoint the record was extracted from.
func (t *Topology) DerivedFrom() string { return t.derivedFrom }

// Components in canonical (name) order.
func (t *Topology) Components() []TopologyComponent { return t.components }

// Component looks one up by name.
func (t *Topology) Component(name string) *TopologyComponent { return t.byName[name] }

// Tensors counts every key across every component — the number a refusal
// message quotes when it says a candidate carries 638 keys and the topology 535.
func (t *Topology) Tensors() int {
	total := 0
	for at := range t.components {
		total += len(t.components[at].tensors)
	}
	return total
}

// Digest is the SHA-256 of the canonical rendering; Stamp is handle + digest.
func (t *Topology) Digest() string { return digestOf(t.canonical()) }

func (t *Topology) canonical() string {
	var out strings.Builder
	out.WriteString(TopologyFormat)
	fmt.Fprintf(&out, "\nname=%s\nversion=%d\ndtype=%s\n",
		t.handle.Name, t.handle.Version, t.dtype)
	for at := range t.components {
		component := &t.components[at]
		fmt.Fprintf(&out, "component name=%s role=%s\n", component.Name, component.Role)
		for _, key := range component.keys {
			fmt.Fprintf(&out, "tensor %s=%s\n", key, shapeCanonical(component.tensors[key]))
		}
		for _, key := range sortedMapKeys(component.Islands) {
			fmt.Fprintf(&out, "island %s=%s\n", key, component.Islands[key])
		}
	}
	return out.String()
}

func shapeCanonical(shape Shape) string {
	parts := make([]string, 0, len(shape))
	for _, dimension := range shape {
		parts = append(parts, fmt.Sprintf("%d", dimension))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// --- the document ----------------------------------------------------------

type rawTopology struct {
	Format      *string           `json:"format"`
	Name        *string           `json:"name"`
	Version     *uint32           `json:"version"`
	Description string            `json:"description"`
	DerivedFrom string            `json:"derived_from"`
	Dtype       string            `json:"dtype"`
	Components  []rawTopologyPart `json:"components"`
	Digest      string            `json:"digest"`
}

type rawTopologyPart struct {
	Name    string              `json:"name"`
	Role    string              `json:"role"`
	Islands map[string]string   `json:"islands,omitempty"`
	Tensors map[string][]uint64 `json:"tensors"`
}

// ParseTopology reads and validates one topology record.
//
// The `digest` field is SELF-PINNING: a document that carries one and does not
// hash to it refuses. Documents are generated, so a hand-edit that changes what
// a topology MEANS without regenerating is exactly the failure this catches.
func ParseTopology(document []byte) (*Topology, error) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.DisallowUnknownFields()
	var raw rawTopology
	if err := decoder.Decode(&raw); err != nil {
		return nil, refuse("json", "%v", err)
	}
	if raw.Format == nil || *raw.Format != TopologyFormat {
		return nil, refuse("format", "not a %s document", TopologyFormat)
	}
	if raw.Name == nil || raw.Version == nil {
		return nil, refuse("identity", "a topology record is named and versioned")
	}
	handle, err := ParseHandle(fmt.Sprintf("%s@%d", *raw.Name, *raw.Version))
	if err != nil {
		return nil, err
	}
	if len(raw.Components) == 0 {
		return nil, refuse("no-components", "%s declares no component", handle)
	}
	if raw.Dtype == "" {
		return nil, refuse("dtype", "%s records no reference dtype", handle)
	}
	topology := &Topology{
		handle: handle, description: raw.Description, derivedFrom: raw.DerivedFrom,
		dtype: raw.Dtype, byName: map[string]*TopologyComponent{},
	}
	for _, part := range raw.Components {
		if part.Name == "" && len(raw.Components) > 1 {
			return nil, refuse("component", "%s has an unnamed component beside others", handle)
		}
		if len(part.Tensors) == 0 {
			return nil, refuse("component", "%s component %q is empty", handle, part.Name)
		}
		if _, duplicate := topology.byName[part.Name]; duplicate {
			return nil, refuse("component", "%s declares %q twice", handle, part.Name)
		}
		component := TopologyComponent{
			Name: part.Name, Role: ComponentRole(part.Role),
			Islands: part.Islands, tensors: map[string]Shape{},
		}
		for key, shape := range part.Tensors {
			if key == "" {
				return nil, refuse("component", "%s has an empty key", handle)
			}
			for _, dimension := range shape {
				if dimension == 0 {
					return nil, refuse("shape", "%s: %s has a zero dimension", handle, key)
				}
			}
			component.tensors[key] = Shape(shape)
		}
		for key := range component.Islands {
			if _, known := component.tensors[key]; !known {
				return nil, refuse("island", "%s: island %q is not a declared tensor", handle, key)
			}
		}
		component.keys = sortedMapKeys(component.tensors)
		topology.components = append(topology.components, component)
	}
	sort.Slice(topology.components, func(i, j int) bool {
		return topology.components[i].Name < topology.components[j].Name
	})
	for at := range topology.components {
		topology.byName[topology.components[at].Name] = &topology.components[at]
	}
	if raw.Digest != "" && raw.Digest != topology.Digest() {
		return nil, refuse("digest",
			"%s carries digest %s but hashes to %s — the record was edited without "+
				"being regenerated", handle, raw.Digest, topology.Digest())
	}
	return topology, nil
}

// Render writes the document form, digest included. The extractor writes this;
// nothing else may author one.
func (t *Topology) Render() ([]byte, error) {
	raw := rawTopology{
		Format: stringPtr(TopologyFormat), Name: stringPtr(t.handle.Name),
		Version: uint32Ptr(t.handle.Version), Description: t.description,
		DerivedFrom: t.derivedFrom, Dtype: t.dtype, Digest: t.Digest(),
	}
	for at := range t.components {
		component := &t.components[at]
		tensors := make(map[string][]uint64, len(component.tensors))
		for key, shape := range component.tensors {
			tensors[key] = shape
		}
		raw.Components = append(raw.Components, rawTopologyPart{
			Name: component.Name, Role: string(component.Role),
			Islands: component.Islands, Tensors: tensors,
		})
	}
	return json.MarshalIndent(raw, "", " ")
}

func stringPtr(value string) *string { return &value }
func uint32Ptr(value uint32) *uint32 { return &value }

// --- extraction ------------------------------------------------------------

// TopologyFromHeaders derives a topology record MECHANICALLY from one reference
// checkpoint's headers. Nothing below is authored: components come from the
// members' own directories, shards flatten because they share one, the
// dominant dtype is a count and the islands are what disagrees with it.
//
// A DIRECTORY IS NOT ALWAYS A COMPONENT. Shards flatten because their key sets
// are DISJOINT — that, and not the shared folder, is what makes them one
// network. A flat directory holding several networks side by side is the other
// case and it is real: TRELLIS.2's `ckpts/` ships three different DiTs (and two
// exact duplicates of two of them) with one identical 640-key name set and
// `input_layer.weight` at [1536, 8] / [1536, 32] / [1536, 64]. Merging those
// into one map would either refuse the reference checkpoint or, for the exact
// duplicates, silently describe five members as one — and the engine would then
// refuse the tree as RefusalDuplicate, because two members may never be
// assigned to one component. So members are partitioned by DISJOINTNESS: a
// member joins a component only when it adds keys nobody in that component
// already covers, which is exactly the rule matchLayout enforces at the other
// end. Directories that yield one component keep the directory's name; a split
// one names each component after its first member's file stem.
func TopologyFromHeaders(name string, version uint32, derivedFrom string, files []ArtifactFile) (*Topology, error) {
	handle, err := ParseHandle(fmt.Sprintf("%s@%d", name, version))
	if err != nil {
		return nil, err
	}
	grouped := map[string][]ArtifactFile{}
	for _, file := range files {
		directory := path.Dir(file.Path)
		if directory == "." {
			directory = ""
		}
		grouped[directory] = append(grouped[directory], file)
	}
	// The dominant dtype is the DENOISER'S, because the denoiser is what a
	// serve lane executes and what its dtype spelling has always meant. Whole-
	// tree tallies lie in both directions on real trees: LTX-2.3 and Z-Image
	// carry F32 text encoders that outweigh their bf16 denoisers in BYTES, and
	// minimax-h3's serving tree carries F32 autoencoders that outnumber its
	// 66 GB bf16 transformer in KEYS. Within the chosen scope bytes decide
	// (a 4 KB norm table is not the peer of a 500 MB weight), with a key-count
	// fallback for members that bank no lengths (GGUF directories carry none).
	// A tree with no denoiser-role directory measures over everything, as
	// before. Whatever wins, every key elsewhere that disagrees is an island —
	// which is exactly how a reference-tolerant plain rule stays able to admit
	// the reference itself.
	dtype := dominantDtype(files, func(directory string) bool {
		return roleOfDirectory(directory) == RoleDenoiser
	})
	if dtype == "" {
		dtype = dominantDtype(files, func(string) bool { return true })
	}
	topology := &Topology{handle: handle, derivedFrom: derivedFrom,
		dtype: dtype, byName: map[string]*TopologyComponent{}}
	for _, directory := range sortedMapKeys(grouped) {
		parts := partitionMembers(grouped[directory])
		for _, part := range parts {
			component := TopologyComponent{
				Name: componentName(directory), Role: roleOfDirectory(directory),
				tensors: map[string]Shape{}, Islands: map[string]string{},
			}
			if len(parts) > 1 {
				component.Name = memberStem(part[0].Path)
			}
			for _, file := range part {
				for _, tensor := range file.Tensors {
					component.tensors[tensor.Name] = Shape(tensor.Shape).clone()
					if tensor.Dtype != topology.dtype {
						component.Islands[tensor.Name] = tensor.Dtype
					}
				}
			}
			if len(component.Islands) == 0 {
				component.Islands = nil
			}
			component.keys = sortedMapKeys(component.tensors)
			topology.components = append(topology.components, component)
		}
	}
	sort.Slice(topology.components, func(i, j int) bool {
		return topology.components[i].Name < topology.components[j].Name
	})
	for at := range topology.components {
		name := topology.components[at].Name
		if _, duplicate := topology.byName[name]; duplicate {
			return nil, refuse("component", "%s: two components both name %q", handle, name)
		}
		topology.byName[name] = &topology.components[at]
	}
	return topology, nil
}

// partitionMembers splits one directory's members into components.
//
// TWO FACTS DECIDE IT, and neither alone is enough.
//
// The FILE NAME says which members were shipped as one sharded artifact:
// `model-00003-of-00026`, `layers-7`, `diffusion_pytorch_model`. Digit runs are
// the shard index, so a stem with every run replaced by `#` is the artifact's
// name — which is what separates TRELLIS.2's `shape_enc_next_dc_f16c32_fp16`
// from its `shape_dec_...` twin, two disjoint networks a keys-only rule would
// happily have fused into one component.
//
// DISJOINTNESS says which of those actually fit in one map. Shards never repeat
// a key, so a whole family collapses to one component; two copies of one
// network shipped side by side at different resolutions do repeat every key, so
// they open separate components — which is required, because the engine may
// never assign two members to one component (RefusalDuplicate).
//
// Path order makes it deterministic; the caller's file order does not.
func partitionMembers(members []ArtifactFile) [][]ArtifactFile {
	ordered := append([]ArtifactFile(nil), members...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Path < ordered[j].Path })
	var families []string
	byFamily := map[string][]ArtifactFile{}
	for _, member := range ordered {
		family := shardFamily(member.Path)
		if _, seen := byFamily[family]; !seen {
			families = append(families, family)
		}
		byFamily[family] = append(byFamily[family], member)
	}
	var parts [][]ArtifactFile
	for _, family := range families {
		var covered []map[string]bool
		first := len(parts)
		for _, member := range byFamily[family] {
			placed := false
			for at := first; at < len(parts); at++ {
				if overlaps(covered[at-first], member) {
					continue
				}
				for _, tensor := range member.Tensors {
					covered[at-first][tensor.Name] = true
				}
				parts[at] = append(parts[at], member)
				placed = true
				break
			}
			if placed {
				continue
			}
			keys := map[string]bool{}
			for _, tensor := range member.Tensors {
				keys[tensor.Name] = true
			}
			parts = append(parts, []ArtifactFile{member})
			covered = append(covered, keys)
		}
	}
	return parts
}

func overlaps(covered map[string]bool, member ArtifactFile) bool {
	for _, tensor := range member.Tensors {
		if covered[tensor.Name] {
			return true
		}
	}
	return false
}

// shardFamily is a member's artifact name: its stem with every run of digits
// replaced, so `model-00001-of-00026`, `model-00026-of-00026` and `layers-7`
// all name the artifact they are a shard of.
func shardFamily(memberPath string) string {
	var out strings.Builder
	digits := false
	for _, character := range memberStem(memberPath) {
		if character >= '0' && character <= '9' {
			if !digits {
				out.WriteByte('#')
			}
			digits = true
			continue
		}
		digits = false
		out.WriteRune(character)
	}
	return out.String()
}

// memberStem is a split component's name: the member's file name without its
// container extension. `ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` names
// the component `ss_flow_img_dit_1_3B_64_bf16`.
func memberStem(memberPath string) string {
	base := path.Base(memberPath)
	return strings.TrimSuffix(base, path.Ext(base))
}

func componentName(directory string) string {
	if directory == "" {
		return ""
	}
	return path.Base(directory)
}

func dominantDtype(files []ArtifactFile, inScope func(directory string) bool) string {
	bytes := map[string]uint64{}
	keys := map[string]uint64{}
	total := uint64(0)
	for _, file := range files {
		directory := path.Dir(file.Path)
		if directory == "." {
			directory = ""
		}
		if !inScope(directory) {
			continue
		}
		for _, tensor := range file.Tensors {
			bytes[tensor.Dtype] += tensor.Length
			keys[tensor.Dtype]++
			total += tensor.Length
		}
	}
	weights := bytes
	if total == 0 {
		weights = keys
	}
	best := ""
	bestWeight := uint64(0)
	first := true
	for _, dtype := range sortedMapKeys(weights) {
		if first || weights[dtype] > bestWeight {
			best, bestWeight, first = dtype, weights[dtype], false
		}
	}
	return best
}
