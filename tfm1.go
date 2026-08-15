package hashrepo

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// TFM1 is the TensorFS canonical snapshot manifest (spec/v1/TFM1.md). This
// side is Tensorhub's strict, non-authoring verifier: it decodes and refuses,
// never encodes. Rust authors the canonical bytes and the shared vectors.

const (
	// TFM1MaxPathBytes bounds one entry path and one symlink target.
	TFM1MaxPathBytes = 4096
	// TFM1MaxFileRecords mirrors the planner's bounded object cardinality.
	TFM1MaxFileRecords = 1_000_000

	tfm1MinEntryBytes  = 4 + 1 + 1
	tfm1MinRecordBytes = 1 + 8

	tfm1KindDirectory = 1
	tfm1KindFile      = 2
	tfm1KindSymlink   = 3
	tfm1KindHardlink  = 4
	tfm1RecordData    = 1
	tfm1RecordHole    = 2
)

// TFM1EntryKind is the wire entry discriminant.
type TFM1EntryKind uint8

const (
	TFM1Directory TFM1EntryKind = tfm1KindDirectory
	TFM1File      TFM1EntryKind = tfm1KindFile
	TFM1Symlink   TFM1EntryKind = tfm1KindSymlink
	TFM1Hardlink  TFM1EntryKind = tfm1KindHardlink
)

// The closed planner provenance vocabulary. Readers reconstruct solely from
// the ordered records; the planner name is provenance and is never executed.
const (
	TFM1PlannerSafetensorsV1 = "safetensors-v1"
	TFM1PlannerGgufV1        = "gguf-v1"
	TFM1PlannerRawFixed64mV1 = "raw-fixed-64m-v1"
)

// TFM1Record is one Data or Hole run of a regular file.
type TFM1Record struct {
	Hole   bool
	Digest Ref // zero for holes
	Length uint64
}

// TFM1Entry is one decoded manifest entry. Field validity follows Kind.
type TFM1Entry struct {
	Path        string
	Kind        TFM1EntryKind
	Executable  bool
	Planner     string
	LogicalSize uint64
	Records     []TFM1Record
	Target      string
	Ordinal     uint64
}

// TFM1Snapshot is one strictly decoded manifest.
type TFM1Snapshot struct {
	// SnapshotID is the lowercase hex SHA-256 of the canonical bytes.
	SnapshotID string
	// Parent is the explicit parent snapshot id, or "" when absent.
	Parent  string
	Entries []TFM1Entry
}

// OrdinalPath resolves a hardlink ordinal to its carrier path. Ordinals count
// regular-file entries in path order, starting at zero.
func (s TFM1Snapshot) OrdinalPath(ordinal uint64) (string, bool) {
	var assigned uint64
	for _, entry := range s.Entries {
		if entry.Kind != TFM1File {
			continue
		}
		if assigned == ordinal {
			return entry.Path, true
		}
		assigned++
	}
	return "", false
}

// TFM1Error is a strict decode refusal carrying the stable kebab-case reason
// label shared byte-for-byte with the Rust implementation and the vectors.
type TFM1Error struct {
	reason string
}

func (e *TFM1Error) Error() string { return "tfm1: " + e.reason }

// Reason returns the cross-language refusal label.
func (e *TFM1Error) Reason() string { return e.reason }

func tfm1Refuse(reason string) error { return &TFM1Error{reason: reason} }

type tfm1Reader struct {
	data []byte
	pos  int
}

func (r *tfm1Reader) take(count int) ([]byte, error) {
	if count > len(r.data)-r.pos {
		return nil, tfm1Refuse("truncated")
	}
	slice := r.data[r.pos : r.pos+count]
	r.pos += count
	return slice, nil
}

func (r *tfm1Reader) takeU8() (byte, error) {
	taken, err := r.take(1)
	if err != nil {
		return 0, err
	}
	return taken[0], nil
}

func (r *tfm1Reader) takeU32() (uint32, error) {
	taken, err := r.take(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(taken), nil
}

func (r *tfm1Reader) takeU64() (uint64, error) {
	taken, err := r.take(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(taken), nil
}

func (r *tfm1Reader) remaining() uint64 {
	return uint64(len(r.data) - r.pos)
}

var tfm1WindowsReserved = []string{
	"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5",
	"com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5",
	"lpt6", "lpt7", "lpt8", "lpt9",
}

func validateTFM1Path(path string) error {
	if path == "" || len(path) > TFM1MaxPathBytes {
		return tfm1Refuse("path-length")
	}
	if !norm.NFC.IsNormalString(path) {
		return tfm1Refuse("path-not-nfc")
	}
	if strings.HasPrefix(path, "/") {
		return tfm1Refuse("path-absolute")
	}
	for _, segment := range strings.Split(path, "/") {
		if segment == "" {
			return tfm1Refuse("path-empty-segment")
		}
		if segment == "." || segment == ".." {
			return tfm1Refuse("path-dot-segment")
		}
		for _, character := range segment {
			if unicode.IsControl(character) ||
				strings.ContainsRune(`\:*?"<>|`, character) {
				return tfm1Refuse("path-forbidden-character")
			}
		}
		if strings.HasSuffix(segment, ".") || strings.HasSuffix(segment, " ") {
			return tfm1Refuse("path-trailing-dot-space")
		}
		stem := segment
		if dot := strings.IndexByte(segment, '.'); dot >= 0 {
			stem = segment[:dot]
		}
		if len(stem) <= 4 {
			folded := asciiFold(stem)
			for _, reserved := range tfm1WindowsReserved {
				if folded == reserved {
					return tfm1Refuse("path-windows-reserved")
				}
			}
		}
	}
	return nil
}

func validateTFM1SymlinkTarget(target string) error {
	if target == "" || len(target) > TFM1MaxPathBytes {
		return tfm1Refuse("symlink-target")
	}
	for _, character := range target {
		if unicode.IsControl(character) {
			return tfm1Refuse("symlink-target")
		}
	}
	return nil
}

func validateTFM1Records(records []TFM1Record, logicalSize uint64) error {
	if len(records) > TFM1MaxFileRecords {
		return tfm1Refuse("record-limit")
	}
	var sum uint64
	previousWasHole := false
	for _, record := range records {
		if record.Length == 0 {
			return tfm1Refuse("zero-length-record")
		}
		if record.Hole {
			if previousWasHole {
				return tfm1Refuse("adjacent-holes")
			}
			previousWasHole = true
		} else {
			if record.Length > uint64(MaxChunkSize) {
				return tfm1Refuse("data-too-large")
			}
			previousWasHole = false
		}
		if sum+record.Length < sum {
			return tfm1Refuse("length-sum-mismatch")
		}
		sum += record.Length
	}
	if sum != logicalSize {
		return tfm1Refuse("length-sum-mismatch")
	}
	return nil
}

// tfm1CaseFold mirrors the Rust full lowercase mapping. The one multi-rune
// context-free lowercase expansion in Unicode is U+0130; every other rune
// lowers through the simple table on both sides.
func tfm1CaseFold(path string) string {
	var folded strings.Builder
	for _, character := range path {
		if character == 'İ' {
			folded.WriteString("i̇")
			continue
		}
		folded.WriteRune(unicode.ToLower(character))
	}
	return folded.String()
}

func validateTFM1Tree(entries []TFM1Entry) error {
	folded := make(map[string]bool, len(entries))
	for _, entry := range entries {
		key := tfm1CaseFold(entry.Path)
		if folded[key] {
			return tfm1Refuse("case-fold-collision")
		}
		folded[key] = true
	}
	directories := make(map[string]bool)
	for _, entry := range entries {
		if entry.Kind == TFM1Directory {
			directories[entry.Path] = true
		}
	}
	for _, entry := range entries {
		if slash := strings.LastIndexByte(entry.Path, '/'); slash >= 0 {
			if !directories[entry.Path[:slash]] {
				return tfm1Refuse("missing-parent-directory")
			}
		}
	}
	return nil
}

// ParseTFM1 strictly decodes one canonical TFM1 manifest. Every structural,
// path, record, ordering, and topology rule refuses with the shared reason
// label; nothing outside the committed vectors' grammar is accepted.
func ParseTFM1(data []byte) (TFM1Snapshot, error) {
	reader := &tfm1Reader{data: data}
	magic, err := reader.take(4)
	if err != nil {
		return TFM1Snapshot{}, err
	}
	if string(magic) != "TFM1" {
		return TFM1Snapshot{}, tfm1Refuse("bad-magic")
	}

	parent := ""
	switch flag, err := reader.takeU8(); {
	case err != nil:
		return TFM1Snapshot{}, err
	case flag == 0:
	case flag == 1:
		bytes, err := reader.take(32)
		if err != nil {
			return TFM1Snapshot{}, err
		}
		parent = hex.EncodeToString(bytes)
	default:
		return TFM1Snapshot{}, tfm1Refuse("parent-flag")
	}

	entryCount, err := reader.takeU64()
	if err != nil {
		return TFM1Snapshot{}, err
	}
	if entryCount > reader.remaining()/tfm1MinEntryBytes {
		return TFM1Snapshot{}, tfm1Refuse("count-exceeds-input")
	}
	entries := make([]TFM1Entry, 0, entryCount)

	var assignedOrdinals uint64
	for index := uint64(0); index < entryCount; index++ {
		pathLength, err := reader.takeU32()
		if err != nil {
			return TFM1Snapshot{}, err
		}
		if pathLength == 0 || pathLength > TFM1MaxPathBytes {
			return TFM1Snapshot{}, tfm1Refuse("path-length")
		}
		pathBytes, err := reader.take(int(pathLength))
		if err != nil {
			return TFM1Snapshot{}, err
		}
		if !utf8.Valid(pathBytes) {
			return TFM1Snapshot{}, tfm1Refuse("path-utf8")
		}
		path := string(pathBytes)
		if err := validateTFM1Path(path); err != nil {
			return TFM1Snapshot{}, err
		}
		if len(entries) > 0 {
			previous := entries[len(entries)-1].Path
			switch {
			case previous == path:
				return TFM1Snapshot{}, tfm1Refuse("duplicate-path")
			case previous > path:
				return TFM1Snapshot{}, tfm1Refuse("entry-order")
			}
		}

		entry := TFM1Entry{Path: path}
		kind, err := reader.takeU8()
		if err != nil {
			return TFM1Snapshot{}, err
		}
		switch kind {
		case tfm1KindDirectory:
			entry.Kind = TFM1Directory
		case tfm1KindFile:
			entry.Kind = TFM1File
			switch executable, err := reader.takeU8(); {
			case err != nil:
				return TFM1Snapshot{}, err
			case executable == 0:
			case executable == 1:
				entry.Executable = true
			default:
				return TFM1Snapshot{}, tfm1Refuse("executable-flag")
			}
			planner, err := reader.takeU8()
			if err != nil {
				return TFM1Snapshot{}, err
			}
			switch planner {
			case 1:
				entry.Planner = TFM1PlannerSafetensorsV1
			case 2:
				entry.Planner = TFM1PlannerGgufV1
			case 3:
				entry.Planner = TFM1PlannerRawFixed64mV1
			default:
				return TFM1Snapshot{}, tfm1Refuse("unknown-planner")
			}
			if entry.LogicalSize, err = reader.takeU64(); err != nil {
				return TFM1Snapshot{}, err
			}
			recordCount, err := reader.takeU64()
			if err != nil {
				return TFM1Snapshot{}, err
			}
			if recordCount > TFM1MaxFileRecords {
				return TFM1Snapshot{}, tfm1Refuse("record-limit")
			}
			if recordCount > reader.remaining()/tfm1MinRecordBytes {
				return TFM1Snapshot{}, tfm1Refuse("count-exceeds-input")
			}
			entry.Records = make([]TFM1Record, 0, recordCount)
			for record := uint64(0); record < recordCount; record++ {
				tag, err := reader.takeU8()
				if err != nil {
					return TFM1Snapshot{}, err
				}
				switch tag {
				case tfm1RecordData:
					digestBytes, err := reader.take(32)
					if err != nil {
						return TFM1Snapshot{}, err
					}
					length, err := reader.takeU64()
					if err != nil {
						return TFM1Snapshot{}, err
					}
					entry.Records = append(entry.Records, TFM1Record{
						Digest: Ref{hex: hex.EncodeToString(digestBytes)},
						Length: length,
					})
				case tfm1RecordHole:
					length, err := reader.takeU64()
					if err != nil {
						return TFM1Snapshot{}, err
					}
					entry.Records = append(entry.Records, TFM1Record{
						Hole:   true,
						Length: length,
					})
				default:
					return TFM1Snapshot{}, tfm1Refuse("unknown-record-tag")
				}
			}
			if err := validateTFM1Records(entry.Records, entry.LogicalSize); err != nil {
				return TFM1Snapshot{}, err
			}
			assignedOrdinals++
		case tfm1KindSymlink:
			entry.Kind = TFM1Symlink
			targetLength, err := reader.takeU32()
			if err != nil {
				return TFM1Snapshot{}, err
			}
			if targetLength > TFM1MaxPathBytes {
				return TFM1Snapshot{}, tfm1Refuse("symlink-target")
			}
			targetBytes, err := reader.take(int(targetLength))
			if err != nil {
				return TFM1Snapshot{}, err
			}
			if !utf8.Valid(targetBytes) {
				return TFM1Snapshot{}, tfm1Refuse("symlink-target")
			}
			entry.Target = string(targetBytes)
			if err := validateTFM1SymlinkTarget(entry.Target); err != nil {
				return TFM1Snapshot{}, err
			}
		case tfm1KindHardlink:
			entry.Kind = TFM1Hardlink
			if entry.Ordinal, err = reader.takeU64(); err != nil {
				return TFM1Snapshot{}, err
			}
			if entry.Ordinal >= assignedOrdinals {
				return TFM1Snapshot{}, tfm1Refuse("hardlink-ordinal")
			}
		default:
			return TFM1Snapshot{}, tfm1Refuse("unknown-entry-kind")
		}
		entries = append(entries, entry)
	}
	if reader.remaining() != 0 {
		return TFM1Snapshot{}, tfm1Refuse("trailing-bytes")
	}
	if err := validateTFM1Tree(entries); err != nil {
		return TFM1Snapshot{}, err
	}

	sum := sha256.Sum256(data)
	return TFM1Snapshot{
		SnapshotID: hex.EncodeToString(sum[:]),
		Parent:     parent,
		Entries:    entries,
	}, nil
}
