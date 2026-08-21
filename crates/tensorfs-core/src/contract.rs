//! SEAM DESCRIPTORS: the declarative input that directs chunking.
//!
//! A document here describes where a fused tensor's seams fall and which named
//! sets of tensors are removable, so the chunker can cut a fused packaging and
//! its split twin into the SAME data objects. It is DATA, not planner code, and
//! it ARRIVES WITH THE CALLER — there is no library and no registry in this
//! crate any more.
//!
//! **What this module no longer does (tensorfs#151).** It used to identify a
//! checkpoint: score an embedded library of contracts against a header and pick
//! a winner. That question — "which layout is this?" — now has exactly one
//! answer-giver, the Go decision engine, which searches v2 topology records and
//! quant rules for the pair whose COMPUTED layout the headers are. Three
//! implementations of that search (here, in Go, and in the Python planner) were
//! three chances to admit a bind that should have been refused, invisible until
//! a pod 500s.
//!
//! Two properties are load-bearing (dedup-invariance memo §4):
//!
//! - Boundaries are a pure function of `(file bytes, contract)`. The store's
//!   contents never enter, so the same file under the same contract chunks
//!   identically on every store, at every time, before and after a GC.
//! - A document is identified by its stamp and that stamp is recorded in the
//!   snapshot: `name@version` for a named one, `sha256:<hex>` — the
//!   digest of the canonical rendering — for an author-constructed custom,
//!   which carries no name at all. Reading a snapshot tells you which layout
//!   directed it without consulting whatever registry happens to be
//!   installed. Bumping `version` is the ONLY way to change a library
//!   contract's meaning; a custom's meaning cannot change without changing
//!   its identity, by construction.
//!
//! Seams never move bytes. They only add cut points inside a fused tensor's
//! own extent, before the 64 MiB grid — so a fused file's objects become the
//! exact union of the split packaging's objects, and load order is untouched.
//!
//! A fusion may be INTERLEAVED (`groups > 1`): MiniMax-H3 fuses its qkv
//! head-major, so the axis reads `q0 k0 v0 q1 k1 v1 …` over 56 heads. That is
//! still an ordered concatenation of contiguous runs, so seams recover it
//! exactly — objects are content-addressed, so the two packagings share bytes
//! even though they read the same runs in different orders. The bound on how
//! fine an interleave may go is [`MIN_SEAM_PART_BYTES`]: below it, the
//! declaration produces no cut points and the case belongs to the adapter
//! (permute) class instead.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::planner::TensorInventory;

/// The one contract document format tag.
pub const CONTRACT_FORMAT: &str = "tensorfs-contract-v1";
/// Bound on a contract name, so a stamp is a bounded manifest field.
pub const MAX_CONTRACT_NAME_BYTES: usize = 64;

// THE EMBEDDED CONTRACT LIBRARY IS GONE (tensorfs#151). `spec/v1/contracts/`
// and the Registry that identified files against it are deleted with the v1
// engine: identification is now the Go decision engine's `Catalog.Stamp`, over
// v2 topology records and quant rules, and there is exactly one of it.
//
// What survives here is MECHANICS. A document parsed by this module tells the
// chunker where a fused tensor's seams fall and tells `adapter`/`compose`
// which bytes a derive may inherit. It decides nothing about admission, it is
// supplied by the caller rather than looked up, and it can no longer answer
// "is this the right checkpoint?" — that question left this crate.

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("contract document is not valid JSON: {0}")]
    Json(String),
    #[error("contract document format is not {CONTRACT_FORMAT}")]
    Format,
    #[error("contract name {0:?} is not a usable contract name")]
    Name(String),
    #[error("contract version must be at least 1")]
    Version,
    #[error("contract name and version are declared together or not at all")]
    Identity,
    #[error("digest stamp {0:?} is not sha256:<64 lowercase hex>")]
    DigestStamp(String),
    #[error("contract dtype {0:?} is not a lowercase torch-style dtype name")]
    Dtype(String),
    #[error("contract declares no tensors")]
    NoTensors,
    #[error("contract pattern {0:?} is malformed")]
    Pattern(String),
    #[error("contract declares {0:?} twice")]
    Duplicate(String),
    #[error("role {0:?} and its pattern do not carry the same holes")]
    RoleHoles(String),
    #[error("fusion declaration on {0:?} is malformed")]
    Fusion(String),
    #[error("permute declaration on {0:?} is malformed")]
    Permute(String),
    #[error("named set {0:?} is malformed")]
    Set(String),
    #[error("the registry holds {0} twice")]
    DuplicateContract(String),
}

impl ContractError {
    /// Stable kebab-case label shared with the language-neutral contract
    /// vectors (`spec/v1/contract-vectors/`), so every implementation's
    /// refusal vocabulary stays aligned by test rather than by discipline.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Json(_) => "json",
            Self::Format => "format",
            Self::Name(_) => "name",
            Self::Version => "version",
            Self::Identity => "identity",
            Self::DigestStamp(_) => "digest-stamp",
            Self::Dtype(_) => "dtype",
            Self::NoTensors => "no-tensors",
            Self::Pattern(_) => "pattern",
            Self::Duplicate(_) => "duplicate",
            Self::RoleHoles(_) => "role-holes",
            Self::Fusion(_) => "fusion",
            Self::Permute(_) => "permute",
            Self::Set(_) => "set",
            Self::DuplicateContract(_) => "duplicate-contract",
        }
    }
}

// ---------------------------------------------------------------------------
// The stamp
// ---------------------------------------------------------------------------

/// One validated contract handle: a bounded `<producer>.<format>` name and a
/// version of at least 1. The fields are private and the constructor is the
/// only way in, so a stamp the manifest cannot encode does not exist.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Handle {
    name: String,
    version: u32,
}

impl Handle {
    pub fn new(name: &str, version: u32) -> Result<Self, ContractError> {
        if !is_contract_name(name) {
            return Err(ContractError::Name(name.to_owned()));
        }
        if version == 0 {
            return Err(ContractError::Version);
        }
        Ok(Self {
            name: name.to_owned(),
            version,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
}

/// Which contract directed one file's chunking. Recorded in the manifest, so
/// a snapshot answers "what layout is this?" without probing and without the
/// caller that produced it.
///
/// Identity has two spellings, deliberately: a NAMED layout is identified
/// `name@version`, because the name is pinned somewhere a reader can check —
/// under v2 that is `spec/v2/`, whose records are digest-pinned and regenerated
/// from real headers. Every other document is identified by the
/// SHA-256 of its canonical rendering, spelled `sha256:<hex>`. A free-text
/// name on an inline document validates nothing and can lie or collide, so a
/// custom carries no name at all.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Stamp {
    /// No contract matched: the plain per-tensor grid directed the chunking.
    #[default]
    None,
    Named(Handle),
    /// A custom contract: [`Contract::digest`] of its nameless document.
    Digest([u8; 32]),
}

impl Stamp {
    /// Builds a stamp, refusing every name a manifest may not carry.
    pub fn named(name: &str, version: u32) -> Result<Self, ContractError> {
        Ok(Self::Named(Handle::new(name, version)?))
    }

    /// Parses the display form: `name@version`, `sha256:<64 hex>`, or `"none"`
    /// for the absent stamp.
    pub fn parse(text: &str) -> Result<Self, ContractError> {
        if text == "none" {
            return Ok(Self::None);
        }
        if let Some(digits) = text.strip_prefix("sha256:") {
            let malformed = || ContractError::DigestStamp(text.to_owned());
            if digits.len() != 64 {
                return Err(malformed());
            }
            let mut digest = [0_u8; 32];
            for (index, pair) in digits.as_bytes().chunks_exact(2).enumerate() {
                let pair = std::str::from_utf8(pair).map_err(|_| malformed())?;
                if pair.bytes().any(|byte| byte.is_ascii_uppercase()) {
                    return Err(malformed());
                }
                digest[index] = u8::from_str_radix(pair, 16).map_err(|_| malformed())?;
            }
            return Ok(Self::Digest(digest));
        }
        let (name, version) = text
            .rsplit_once('@')
            .ok_or_else(|| ContractError::Name(text.to_owned()))?;
        let version = version.parse::<u32>().map_err(|_| ContractError::Version)?;
        Self::named(name, version)
    }

    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::None | Self::Digest(_) => None,
            Self::Named(handle) => Some(handle.name()),
        }
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        match self {
            Self::None | Self::Digest(_) => 0,
            Self::Named(handle) => handle.version(),
        }
    }

    /// The digest of a custom contract's stamp; `None` for the other arms.
    #[must_use]
    pub const fn digest(&self) -> Option<&[u8; 32]> {
        match self {
            Self::Digest(digest) => Some(digest),
            _ => None,
        }
    }
}

impl fmt::Display for Stamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Named(handle) => {
                write!(formatter, "{}@{}", handle.name(), handle.version())
            }
            Self::Digest(digest) => {
                formatter.write_str("sha256:")?;
                for byte in digest {
                    write!(formatter, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}

/// The contract-handle shape, deliberately identical to gen-worker's
/// tensor-layout-contract handles (`<producer>.<format>@<major>`): producer is
/// lowercase alphanumeric, format is lowercase alphanumeric with `.`/`-`/`_`.
/// One spelling per contract, so a stamp round-trips through the manifest byte
/// for byte and reads the same on both sides of the serving boundary.
#[must_use]
pub fn is_contract_name(name: &str) -> bool {
    if name.is_empty() || name.len() > MAX_CONTRACT_NAME_BYTES {
        return false;
    }
    let Some((producer, format)) = name.split_once('.') else {
        return false;
    };
    // The producer segment carries a hyphen because gen-worker's model-type
    // vocabulary does: `hidream-o1`, `flux-2`, `wan-2`. A grammar that cannot
    // spell the producer names the platform already uses is the grammar that
    // is wrong. Leading hyphen still refuses.
    if producer.is_empty()
        || !producer.starts_with(|character: char| {
            character.is_ascii_lowercase() || character.is_ascii_digit()
        })
        || !producer.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
        })
    {
        return false;
    }
    let mut characters = format.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    characters.all(|character| {
        character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || matches!(character, '.' | '-' | '_')
    })
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
enum Piece {
    Literal(String),
    /// One non-negative decimal integer without leading zeros.
    Integer,
}

/// A tensor-name pattern: literal text with `{i}` integer holes. No regex, no
/// backtracking — a hole is anchored by the literal that follows it, which the
/// parser forces to start with a non-digit, so matching is linear and the
/// capture list is unambiguous.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern {
    pieces: Vec<Piece>,
    text: String,
}

impl Pattern {
    pub fn parse(text: &str) -> Result<Self, ContractError> {
        let malformed = || ContractError::Pattern(text.to_owned());
        if text.is_empty() || text.len() > 512 {
            return Err(malformed());
        }
        let mut pieces = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match rest.find("{i}") {
                None => {
                    pieces.push(Piece::Literal(rest.to_owned()));
                    break;
                }
                Some(at) => {
                    if at > 0 {
                        pieces.push(Piece::Literal(rest[..at].to_owned()));
                    } else if matches!(pieces.last(), Some(Piece::Integer)) {
                        // Adjacent holes cannot be separated by any input.
                        return Err(malformed());
                    }
                    pieces.push(Piece::Integer);
                    rest = &rest[at + 3..];
                    if rest.starts_with(|character: char| character.is_ascii_digit()) {
                        // A digit after a hole makes the split ambiguous.
                        return Err(malformed());
                    }
                }
            }
        }
        // `{i}` is the only hole spelling, so a brace surviving in a literal
        // is a hole the parser did not understand rather than literal text.
        if pieces.iter().any(|piece| match piece {
            Piece::Literal(literal) => literal.contains('{') || literal.contains('}'),
            Piece::Integer => false,
        }) {
            return Err(malformed());
        }
        Ok(Self {
            pieces,
            text: text.to_owned(),
        })
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    #[must_use]
    fn holes(&self) -> usize {
        self.pieces
            .iter()
            .filter(|piece| matches!(piece, Piece::Integer))
            .count()
    }

    /// The captured hole values when `name` is exactly this pattern.
    #[must_use]
    pub fn captures(&self, name: &str) -> Option<Vec<u64>> {
        let mut rest = name;
        let mut captured = Vec::new();
        for piece in &self.pieces {
            match piece {
                Piece::Literal(literal) => {
                    rest = rest.strip_prefix(literal.as_str())?;
                }
                Piece::Integer => {
                    let digits = rest
                        .find(|character: char| !character.is_ascii_digit())
                        .unwrap_or(rest.len());
                    let (number, tail) = rest.split_at(digits);
                    if number.is_empty() || (number.len() > 1 && number.starts_with('0')) {
                        return None;
                    }
                    captured.push(number.parse::<u64>().ok()?);
                    rest = tail;
                }
            }
        }
        rest.is_empty().then_some(captured)
    }

    #[must_use]
    pub fn matches(&self, name: &str) -> bool {
        self.captures(name).is_some()
    }

    /// Fills this pattern's holes with `values`, in order.
    #[must_use]
    pub fn instantiate(&self, values: &[u64]) -> Option<String> {
        if values.len() != self.holes() {
            return None;
        }
        let mut filled = String::new();
        let mut next = values.iter();
        for piece in &self.pieces {
            match piece {
                Piece::Literal(literal) => filled.push_str(literal),
                Piece::Integer => filled.push_str(&next.next()?.to_string()),
            }
        }
        Some(filled)
    }
}

// ---------------------------------------------------------------------------
// Fusions and their seams
// ---------------------------------------------------------------------------

/// The smallest byte run a declared seam may produce.
///
/// This is the memo's row-granular rejection made into a number. A file cut
/// entirely at this floor still fits TFM1's 1M-record bound at 1 TiB — larger
/// than any single artifact the no-sharding packaging ruling contemplates —
/// while a KB-scale row shuffle blows through it. A fusion whose runs fall
/// below the floor therefore yields NO cut points: it costs the sharing it
/// would have bought and stays a candidate for the adapter class instead.
///
/// The check is a pure function of the tensor's extent and the declaration, so
/// the fused packaging and the split packaging cross it at exactly the same
/// size — a fusion never degrades on one side only.
///
/// FROZEN for v1 (Paul, 2026-08-17): this constant decides chunk boundaries,
/// so it is part of snapshot identity exactly like `MAX_OBJECT_SIZE`. Moving
/// it makes identical inputs chunk differently across planner versions —
/// silent dedup loss with no red anywhere. It changes only with a format
/// version bump. `seam_floor_is_frozen_for_v1` pins it.
pub const MIN_SEAM_PART_BYTES: u64 = 1024 * 1024;

/// One part of a fused tensor: the role suffix it carries in the split
/// packaging, and its share of one group of the outermost axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionPart {
    pub(crate) role: String,
    pub(crate) share: u64,
}

impl FusionPart {
    /// The role suffix, or the empty string when this part IS the whole
    /// declared role — the split side of an interleaved pair, where the
    /// tensor is one part sliced into `groups` runs.
    #[must_use]
    pub fn role(&self) -> &str {
        &self.role
    }

    #[must_use]
    pub const fn share(&self) -> u64 {
        self.share
    }
}

/// One byte run a fusion cuts a tensor into: its length, and the role suffix
/// that identifies the same bytes in the other packaging.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionRun {
    pub(crate) role: String,
    pub(crate) length: u64,
}

impl FusionRun {
    /// The whole tensor as its own single run: what an undeclared (unfused)
    /// tensor contributes to the role map.
    #[must_use]
    pub const fn whole(length: u64) -> Self {
        Self {
            role: String::new(),
            length,
        }
    }

    #[must_use]
    pub fn role(&self) -> &str {
        &self.role
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }
}

/// A fusion along the OUTERMOST stored axis: `groups` repetitions of an
/// ordered part cycle.
///
/// `groups == 1` is plain stacking — `concat([q, k, v], dim=0)`, the SDXL
/// CLIP-G and DiT case. `groups > 1` is an INTERLEAVE: MiniMax-H3 fuses
/// `qkv_proj [21504, 5376]` as 56 head-major triples (`stack(dim=1)`), so the
/// axis reads `q0 k0 v0 q1 k1 v1 …`. Both are ordered concatenations of
/// contiguous runs — the interleave just has 3x56 of them instead of 3 — so
/// both are exactly recoverable by cut points, and both leave byte ORDER
/// untouched. Which run belongs to which split tensor is carried by the run's
/// role, not by its position, and objects are content-addressed, so the fused
/// file and the split trio share every data object despite reading their runs
/// in different orders.
///
/// Only axis 0 is a byte concatenation. An inner-axis fusion (GPT-2 `c_attn`)
/// is a re-arrangement and belongs to the adapter vocabulary; declaring one
/// here is a parse refusal, not a silent inner cut.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fusion {
    pub(crate) groups: u64,
    pub(crate) parts: Vec<FusionPart>,
}

impl Fusion {
    #[must_use]
    pub fn parts(&self) -> &[FusionPart] {
        &self.parts
    }

    #[must_use]
    pub const fn groups(&self) -> u64 {
        self.groups
    }

    /// True when the parts of one split tensor are scattered through the fused
    /// axis rather than contiguous in it.
    #[must_use]
    pub const fn is_interleaved(&self) -> bool {
        self.groups > 1
    }

    fn units(&self) -> Option<u64> {
        let per_group: u64 = self.parts.iter().map(|part| part.share).sum();
        per_group.checked_mul(self.groups)
    }

    /// The ordered runs this fusion cuts a tensor into, in file order, each
    /// tagged with the role suffix its bytes carry in the split packaging.
    ///
    /// `None` when the declaration cannot apply to this tensor: an outer axis
    /// that does not divide into the declared units, or an extent that is not
    /// a whole number of rows. A contract whose fusion cannot apply does not
    /// match the file at all, so a stamped snapshot always means the
    /// declaration was usable.
    #[must_use]
    pub fn runs(&self, shape: &[u64], byte_length: u64) -> Option<Vec<FusionRun>> {
        let rows = *shape.first()?;
        let units = self.units()?;
        if rows == 0 || units == 0 || !rows.is_multiple_of(units) {
            return None;
        }
        if !byte_length.is_multiple_of(rows) {
            return None;
        }
        let unit_bytes = (byte_length / rows).checked_mul(rows / units)?;
        let mut runs = Vec::with_capacity(usize::try_from(units).ok()?);
        for group in 0..self.groups {
            for part in &self.parts {
                let mut role = String::new();
                if !part.role.is_empty() {
                    role.push('#');
                    role.push_str(&part.role);
                }
                if self.groups > 1 {
                    role.push('@');
                    role.push_str(&group.to_string());
                }
                runs.push(FusionRun {
                    role,
                    length: unit_bytes.checked_mul(part.share)?,
                });
            }
        }
        Some(runs)
    }
}

// ---------------------------------------------------------------------------
// Permutes
// ---------------------------------------------------------------------------

/// One dimension of a permute view: a literal, one axis of the tensor's own
/// shape (optionally divided), or the single inferred dimension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Dim {
    Literal(u64),
    /// `shape[k]`, optionally `shape[k]/n`.
    Axis {
        axis: usize,
        divisor: u64,
    },
    /// `auto`: whatever makes the product come out right. At most one.
    Auto,
}

impl Dim {
    fn parse(text: &str) -> Option<Self> {
        if text == "auto" {
            return Some(Self::Auto);
        }
        if let Ok(literal) = text.parse::<u64>() {
            return (literal > 0).then_some(Self::Literal(literal));
        }
        let (head, divisor) = match text.split_once('/') {
            None => (text, 1),
            Some((head, tail)) => (head, tail.parse::<u64>().ok().filter(|value| *value > 0)?),
        };
        let axis = head
            .strip_prefix("shape[")?
            .strip_suffix(']')?
            .parse::<usize>()
            .ok()?;
        Some(Self::Axis { axis, divisor })
    }
}

/// A GENERALIZED permute: reshape to `view`, permute those axes, reshape
/// back. Plain transpose is the two-axis case; the llama.cpp rope-permute
/// (`reshape(n_head, 2, d/2, …).swapaxes(1, 2)`) is why the primitive has to
/// be this and not `transpose`.
///
/// A permute is the one thing in this vocabulary that MOVES bytes inside a
/// tensor. It is exactly invertible and dtype-preserving — layout, not math —
/// so it is servable from bytes we already hold; it is simply not
/// chunk-shareable, which is why it is declared here rather than as a fusion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Permute {
    pub(crate) view: Vec<Dim>,
    pub(crate) axes: Vec<usize>,
}

impl Permute {
    #[must_use]
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// The concrete view dimensions for a tensor of this shape, or `None`
    /// when the declaration cannot apply to it.
    #[must_use]
    pub fn resolve(&self, shape: &[u64]) -> Option<Vec<u64>> {
        let elements: u64 = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))?;
        let mut resolved = Vec::with_capacity(self.view.len());
        let mut auto = None;
        let mut known = 1_u64;
        for (index, dimension) in self.view.iter().enumerate() {
            let value = match dimension {
                Dim::Literal(literal) => *literal,
                Dim::Axis { axis, divisor } => {
                    let axis = *shape.get(*axis)?;
                    if *divisor == 0 || !axis.is_multiple_of(*divisor) {
                        return None;
                    }
                    axis / divisor
                }
                Dim::Auto => {
                    if auto.replace(index).is_some() {
                        return None;
                    }
                    1
                }
            };
            if value == 0 {
                return None;
            }
            known = known.checked_mul(value)?;
            resolved.push(value);
        }
        match auto {
            None => (known == elements).then_some(resolved),
            Some(index) => {
                if known == 0 || !elements.is_multiple_of(known) {
                    return None;
                }
                resolved[index] = elements / known;
                Some(resolved)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------

/// One declared tensor family inside a contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorPattern {
    pub(crate) role: Pattern,
    pub(crate) pattern: Pattern,
    pub(crate) dtypes: Vec<String>,
    pub(crate) rank: Option<usize>,
    pub(crate) required: bool,
    pub(crate) fusion: Option<Fusion>,
    pub(crate) permute: Option<Permute>,
}

impl TensorPattern {
    #[must_use]
    pub fn role(&self) -> &Pattern {
        &self.role
    }

    #[must_use]
    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    /// The accepted element-type spellings; empty accepts any.
    #[must_use]
    pub fn dtypes(&self) -> &[String] {
        &self.dtypes
    }

    /// The declared number of axes, if the declaration constrains it.
    #[must_use]
    pub const fn rank(&self) -> Option<usize> {
        self.rank
    }

    /// Whether a file must carry this declaration to implement the contract.
    #[must_use]
    pub const fn is_required(&self) -> bool {
        self.required
    }

    #[must_use]
    pub const fn fusion(&self) -> Option<&Fusion> {
        self.fusion.as_ref()
    }

    /// How this layout re-arranges the role's canonical element order, if it
    /// does. Two contracts whose permutes differ for one role serve the same
    /// bytes in a different order: viewable, not chunk-shareable.
    #[must_use]
    pub const fn permute(&self) -> Option<&Permute> {
        self.permute.as_ref()
    }

    #[must_use]
    pub fn accepts(&self, dtype: &str, shape: &[u64]) -> bool {
        if !self.dtypes.is_empty() && !self.dtypes.iter().any(|declared| declared == dtype) {
            return false;
        }
        if self.rank.is_some_and(|rank| rank != shape.len()) {
            return false;
        }
        true
    }
}

/// A layout contract: versioned and named when it ships in the curated
/// library, anonymous — identified by digest alone — when an author
/// constructs it inline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Contract {
    /// `Some` for library documents; `None` for customs, whose identity is
    /// [`Contract::digest`]. Name and version travel together or not at all.
    handle: Option<Handle>,
    description: String,
    /// The serve-side load dtype, torch spelling (`"bfloat16"`,
    /// `"float8_e4m3fn"`, ...). Nothing MATCHES on it — per-tensor `dtypes`
    /// stay the matcher's business; this is what `ctx.lane.dtype` reads.
    dtype: Option<String>,
    tensors: Vec<TensorPattern>,
    sets: BTreeMap<String, Vec<Pattern>>,
}

impl Contract {
    pub fn parse(document: &str) -> Result<Self, ContractError> {
        let raw: RawContract = serde_json::from_str(document)
            .map_err(|error| ContractError::Json(error.to_string()))?;
        raw.validate()
    }

    /// The library name, or `None` for an anonymous custom.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.handle.as_ref().map(Handle::name)
    }

    /// The library version; 0 for an anonymous custom, which has none.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.handle.as_ref().map_or(0, Handle::version)
    }

    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The optional top-level load dtype: what serve-side code loads this
    /// layout's tensors as. `None` when the document does not declare one.
    #[must_use]
    pub fn dtype(&self) -> Option<&str> {
        self.dtype.as_deref()
    }

    #[must_use]
    pub fn tensors(&self) -> &[TensorPattern] {
        &self.tensors
    }

    /// `name@version` for a library document, `sha256:<digest>` for a custom.
    #[must_use]
    pub fn stamp(&self) -> Stamp {
        match &self.handle {
            Some(handle) => Stamp::Named(handle.clone()),
            None => Stamp::Digest(self.digest()),
        }
    }

    /// SHA-256 over this contract's canonical rendering.
    ///
    /// For a LIBRARY document this is the proof behind the `name@version`
    /// promise: an edit to a published document changes the digest and the
    /// pinned-digest test fails until the version is bumped. For a CUSTOM
    /// (nameless) document this digest IS the identity — the stamp a snapshot
    /// records, whitespace-invariant and reproducible on every store.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical().as_bytes());
        hasher.finalize().into()
    }

    /// The canonical rendering is OMISSION-PRESERVING: an absent field emits
    /// no line at all, so the digests of the pre-existing named library are
    /// byte-identical to what they were when `name`/`version` were mandatory.
    fn canonical(&self) -> String {
        let mut text = format!("{CONTRACT_FORMAT}\n");
        if let Some(handle) = &self.handle {
            let _ = std::fmt::Write::write_fmt(
                &mut text,
                format_args!("name={}\nversion={}\n", handle.name(), handle.version()),
            );
        }
        if let Some(dtype) = &self.dtype {
            text.push_str("dtype=");
            text.push_str(dtype);
            text.push('\n');
        }
        for tensor in &self.tensors {
            text.push_str(&format!(
                "tensor role={} pattern={} rank={} required={} dtypes={}",
                tensor.role.as_str(),
                tensor.pattern.as_str(),
                tensor
                    .rank
                    .map_or_else(|| "any".to_owned(), |rank| rank.to_string()),
                tensor.required,
                tensor.dtypes.join(","),
            ));
            if let Some(permute) = &tensor.permute {
                text.push_str(" permute=");
                for dimension in &permute.view {
                    match dimension {
                        Dim::Literal(literal) => text.push_str(&literal.to_string()),
                        Dim::Auto => text.push_str("auto"),
                        Dim::Axis { axis, divisor } => {
                            text.push_str(&format!("shape[{axis}]/{divisor}"));
                        }
                    }
                    text.push(',');
                }
                text.push_str(&format!(":{:?}", permute.axes));
            }
            if let Some(fusion) = &tensor.fusion {
                text.push_str(&format!(" fusion=groups:{},", fusion.groups));
                for part in &fusion.parts {
                    text.push_str(&format!("{}:{},", part.role, part.share));
                }
            }
            text.push('\n');
        }
        for (name, patterns) in &self.sets {
            text.push_str(&format!("set {name}="));
            for pattern in patterns {
                text.push_str(pattern.as_str());
                text.push(',');
            }
            text.push('\n');
        }
        text
    }

    /// The declared entry governing a tensor name: the FIRST whose pattern
    /// matches, so overlapping declarations resolve in declaration order
    /// rather than by search luck.
    #[must_use]
    pub fn entry_for(&self, name: &str) -> Option<&TensorPattern> {
        self.tensors
            .iter()
            .find(|tensor| tensor.pattern.matches(name))
    }

    /// The seam cut points inside one tensor, as byte offsets relative to the
    /// tensor's own start, excluding 0 and the tensor's length.
    ///
    /// Empty when no fusion is declared, when the declaration cannot apply, or
    /// when it would produce runs below [`MIN_SEAM_PART_BYTES`]. All three are
    /// the same outcome by design: the tensor grids plainly and forfeits the
    /// sharing, which is a cost, never a correctness question.
    #[must_use]
    pub fn seam_offsets(&self, name: &str, shape: &[u64], byte_length: u64) -> Vec<u64> {
        let Some(runs) = self.runs_of(name, shape, byte_length) else {
            return Vec::new();
        };
        let mut offsets = Vec::with_capacity(runs.len() - 1);
        let mut cursor = 0_u64;
        for run in &runs[..runs.len() - 1] {
            cursor += run.length;
            offsets.push(cursor);
        }
        offsets
    }

    /// The declared byte runs of one tensor, or `None` when it is not cut.
    #[must_use]
    pub fn runs_of(&self, name: &str, shape: &[u64], byte_length: u64) -> Option<Vec<FusionRun>> {
        let runs = self
            .entry_for(name)?
            .fusion
            .as_ref()?
            .runs(shape, byte_length)?;
        if runs.iter().any(|run| run.length < MIN_SEAM_PART_BYTES) {
            return None;
        }
        Some(runs)
    }

    /// The tensors of `inventory` that belong to the named set, in file order.
    #[must_use]
    pub fn set_members<'a>(&self, set: &str, inventory: &'a TensorInventory) -> Vec<&'a str> {
        let Some(patterns) = self.sets.get(set) else {
            return Vec::new();
        };
        inventory
            .tensors()
            .iter()
            .filter(|tensor| {
                patterns
                    .iter()
                    .any(|pattern| pattern.matches(tensor.name()))
            })
            .map(|tensor| tensor.name())
            .collect()
    }

    #[must_use]
    pub fn set_names(&self) -> Vec<&str> {
        self.sets.keys().map(String::as_str).collect()
    }

    // `matches` IS DELETED (tensorfs#151). Header-vs-document matching now
    // happens once, in Go, against a computed layout.
}

// ---------------------------------------------------------------------------
// THE REGISTRY AND ITS DETECTION ARE DELETED (tensorfs#151)
//
// `Registry::detect` scored every library contract against a file's inventory
// and picked a winner. That is IDENTIFICATION, and it now happens exactly once
// in the whole system: the Go engine stamps a checkpoint by searching the v2
// catalog for the (topology, quant) pair whose COMPUTED layout the headers are.
// Three implementations of that search — here, in Go, and in the Python
// planner — were three chances to admit a bind that should have been refused.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Document parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawContract {
    format: String,
    /// Present on library documents; ABSENT on author-constructed customs,
    /// whose identity is the content digest. The two travel together.
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    description: String,
    /// Optional top-level load dtype, torch spelling. ADDITIVE: absent from
    /// the document means absent from the canonical rendering, so documents
    /// that predate the field keep their digests.
    #[serde(default)]
    dtype: Option<String>,
    tensors: Vec<RawTensor>,
    #[serde(default)]
    sets: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    role: String,
    pattern: String,
    #[serde(default)]
    dtypes: Vec<String>,
    #[serde(default)]
    rank: Option<usize>,
    #[serde(default = "yes")]
    required: bool,
    #[serde(default)]
    fusion: Option<RawFusion>,
    #[serde(default)]
    permute: Option<RawPermute>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermute {
    /// Dimensions to reshape to before permuting: a positive integer (as a
    /// number or a string), `shape[k]`, `shape[k]/n`, or `auto` (at most one).
    view: Vec<serde_json::Value>,
    axes: Vec<usize>,
}

const fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFusion {
    /// Declared and checked rather than assumed: only the outermost axis is a
    /// byte concatenation, so `axis` exists to make a wrong declaration a
    /// refusal instead of a silent inner-axis cut.
    axis: usize,
    /// Repetitions of the part cycle along that axis. 1 is plain stacking;
    /// more is an interleave (H3's head-major qkv is 56).
    #[serde(default = "one")]
    groups: u64,
    parts: Vec<RawFusionPart>,
}

const fn one() -> u64 {
    1
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFusionPart {
    role: String,
    share: u64,
}

impl RawContract {
    fn validate(self) -> Result<Contract, ContractError> {
        if self.format != CONTRACT_FORMAT {
            return Err(ContractError::Format);
        }
        let handle = match (self.name, self.version) {
            (None, None) => None,
            (Some(name), Some(version)) => Some(Handle::new(&name, version)?),
            _ => return Err(ContractError::Identity),
        };
        if let Some(dtype) = &self.dtype {
            // A spelling check, not an enum: new torch dtypes must not need a
            // parser change. `getattr(torch, dtype)` is the consumer.
            let usable = !dtype.is_empty()
                && dtype.len() <= 32
                && dtype.chars().all(|character| {
                    character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
                });
            if !usable {
                return Err(ContractError::Dtype(dtype.clone()));
            }
        }
        if self.tensors.is_empty() {
            return Err(ContractError::NoTensors);
        }

        let mut patterns = HashSet::new();
        let mut roles = HashSet::new();
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for raw in self.tensors {
            let pattern = Pattern::parse(&raw.pattern)?;
            let role = Pattern::parse(&raw.role)?;
            if role.holes() != pattern.holes() {
                return Err(ContractError::RoleHoles(raw.role));
            }
            if !patterns.insert(raw.pattern.clone()) {
                return Err(ContractError::Duplicate(raw.pattern));
            }
            if !roles.insert(raw.role.clone()) {
                return Err(ContractError::Duplicate(raw.role));
            }
            let fusion = match raw.fusion {
                None => None,
                Some(raw_fusion) => {
                    let malformed = || ContractError::Fusion(raw.pattern.clone());
                    // Only the outer axis concatenates; a single part in a
                    // single group is the whole tensor, which is not a fusion.
                    if raw_fusion.axis != 0
                        || raw_fusion.groups == 0
                        || raw_fusion.parts.is_empty()
                        || (raw_fusion.groups == 1 && raw_fusion.parts.len() < 2)
                    {
                        return Err(malformed());
                    }
                    let mut part_roles = HashSet::new();
                    let mut parts = Vec::with_capacity(raw_fusion.parts.len());
                    for part in raw_fusion.parts {
                        if part.share == 0 || !part_roles.insert(part.role.clone()) {
                            return Err(malformed());
                        }
                        // An unnamed part is only meaningful as the sole part
                        // of an interleaved slice: it IS the declared role.
                        if part.role.is_empty() && part_roles.len() > 1 {
                            return Err(malformed());
                        }
                        parts.push(FusionPart {
                            role: part.role,
                            share: part.share,
                        });
                    }
                    if parts.iter().any(|part| part.role.is_empty()) && parts.len() > 1 {
                        return Err(malformed());
                    }
                    Some(Fusion {
                        groups: raw_fusion.groups,
                        parts,
                    })
                }
            };
            let permute = match raw.permute {
                None => None,
                Some(raw_permute) => {
                    let malformed = || ContractError::Permute(raw.pattern.clone());
                    // A permute inside a fused tensor would mean its seam runs
                    // are NOT the split packaging's bytes, which is the one
                    // thing a seam declaration promises.
                    if fusion.is_some() {
                        return Err(malformed());
                    }
                    if raw_permute.view.len() < 2
                        || raw_permute.axes.len() != raw_permute.view.len()
                    {
                        return Err(malformed());
                    }
                    let mut seen = vec![false; raw_permute.axes.len()];
                    for axis in &raw_permute.axes {
                        if *axis >= seen.len() || seen[*axis] {
                            return Err(malformed());
                        }
                        seen[*axis] = true;
                    }
                    if raw_permute
                        .axes
                        .iter()
                        .enumerate()
                        .all(|(at, axis)| at == *axis)
                    {
                        // The identity permute is the absence of one; two
                        // spellings of "canonical" would break comparison.
                        return Err(malformed());
                    }
                    let mut view = Vec::with_capacity(raw_permute.view.len());
                    for dimension in &raw_permute.view {
                        let parsed = match dimension {
                            serde_json::Value::Number(number) => {
                                number.as_u64().filter(|value| *value > 0).map(Dim::Literal)
                            }
                            serde_json::Value::String(text) => Dim::parse(text),
                            _ => None,
                        };
                        view.push(parsed.ok_or_else(malformed)?);
                    }
                    Some(Permute {
                        view,
                        axes: raw_permute.axes,
                    })
                }
            };
            tensors.push(TensorPattern {
                role,
                pattern,
                dtypes: raw.dtypes,
                rank: raw.rank,
                required: raw.required,
                fusion,
                permute,
            });
        }

        let mut sets = BTreeMap::new();
        for (name, members) in self.sets {
            if members.is_empty() {
                return Err(ContractError::Set(name));
            }
            let mut compiled = Vec::with_capacity(members.len());
            for member in members {
                compiled.push(Pattern::parse(&member)?);
            }
            sets.insert(name, compiled);
        }

        Ok(Contract {
            handle,
            description: self.description,
            dtype: self.dtype,
            tensors,
            sets,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(document: &str) -> Contract {
        Contract::parse(document).unwrap()
    }

    const FUSED: &str = r#"{
        "format": "tensorfs-contract-v1",
        "name": "test.fused",
        "version": 2,
        "tensors": [
            {"role": "layers.{i}.attn.qkv", "pattern": "layers.{i}.qkv.weight", "rank": 2,
             "fusion": {"axis": 0, "parts": [{"role": "q", "share": 2}, {"role": "k", "share": 1},
                                             {"role": "v", "share": 1}]}},
            {"role": "layers.{i}.mlp", "pattern": "layers.{i}.mlp.weight", "required": false}
        ],
        "sets": {"adaln": ["layers.{i}.adaln.weight"]}
    }"#;

    #[test]
    fn a_pattern_captures_and_instantiates_its_holes() {
        let pattern = Pattern::parse("model.layers.{i}.attn.{i}.weight").unwrap();
        assert_eq!(
            pattern.captures("model.layers.12.attn.0.weight"),
            Some(vec![12, 0])
        );
        assert_eq!(
            pattern.instantiate(&[12, 0]).unwrap(),
            "model.layers.12.attn.0.weight"
        );
        assert_eq!(pattern.captures("model.layers.x.attn.0.weight"), None);
        // No leading zeros: one integer has exactly one spelling.
        assert_eq!(pattern.captures("model.layers.01.attn.0.weight"), None);
        assert_eq!(
            pattern.captures("model.layers.12.attn.0.weight.extra"),
            None
        );
    }

    #[test]
    fn ambiguous_patterns_refuse_at_parse() {
        assert!(Pattern::parse("layers.{i}{i}.weight").is_err());
        assert!(Pattern::parse("layers.{i}0.weight").is_err());
        assert!(Pattern::parse("layers.{j}.weight").is_err());
        assert!(Pattern::parse("").is_err());
    }

    const INTERLEAVED: &str = r#"{
        "format": "tensorfs-contract-v1",
        "name": "test.interleaved",
        "version": 1,
        "tensors": [
            {"role": "b.{i}.qkv", "pattern": "b.{i}.qkv.weight", "rank": 2,
             "fusion": {"axis": 0, "groups": 4,
                        "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1},
                                  {"role": "v", "share": 1}]}},
            {"role": "b.{i}.qkv#q", "pattern": "b.{i}.q.weight", "rank": 2, "required": false,
             "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}}
        ]
    }"#;

    const ONE_PART: &str = r#"{
        "format": "tensorfs-contract-v1",
        "name": "test.one-part",
        "version": 1,
        "tensors": [
            {"role": "b.{i}.qkv", "pattern": "b.{i}.qkv.weight",
             "fusion": {"axis": 0, "parts": [{"role": "q", "share": 1}]}}
        ]
    }"#;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn seam_offsets_cut_the_outer_axis_at_declared_fractions() {
        let contract = contract(FUSED);
        // 8 rows of 1 MiB; shares 2:1:1 give 4/2/2 rows.
        assert_eq!(
            contract.seam_offsets("layers.3.qkv.weight", &[8, 4], 8 * MIB),
            vec![4 * MIB, 6 * MIB]
        );
        // A row count that does not divide the shares is not cuttable.
        assert!(
            contract
                .seam_offsets("layers.3.qkv.weight", &[6, 4], 6 * MIB)
                .is_empty()
        );
        // A tensor with no declared fusion is never cut.
        assert!(
            contract
                .seam_offsets("layers.3.mlp.weight", &[8, 4], 8 * MIB)
                .is_empty()
        );
    }

    #[test]
    fn an_interleaved_fusion_cuts_every_group_and_names_both_sides_alike() {
        let contract = contract(INTERLEAVED);
        // The H3 shape in miniature: 4 groups of (q, k, v), one run each.
        let fused = contract
            .runs_of("b.0.qkv.weight", &[12, 2], 12 * MIB)
            .expect("the fusion applies");
        assert_eq!(fused.len(), 12);
        assert!(fused.iter().all(|run| run.length() == MIB));
        let roles: Vec<&str> = fused.iter().map(FusionRun::role).collect();
        assert_eq!(&roles[..4], ["#q@0", "#k@0", "#v@0", "#q@1"]);
        assert_eq!(
            contract.seam_offsets("b.0.qkv.weight", &[12, 2], 12 * MIB),
            (1..12).map(|part| part * MIB).collect::<Vec<_>>()
        );

        // The split twin declares the SAME runs under the same roles, which is
        // what makes the two packagings share objects: `b.0.qkv#q` + `@g` on
        // this side, `b.0.qkv` + `#q@g` on the fused side.
        let split = contract
            .runs_of("b.0.q.weight", &[4, 2], 4 * MIB)
            .expect("the slice applies");
        assert_eq!(split.len(), 4);
        assert!(split.iter().all(|run| run.length() == MIB));
        assert_eq!(
            split.iter().map(FusionRun::role).collect::<Vec<_>>(),
            ["@0", "@1", "@2", "@3"]
        );
    }

    #[test]
    fn runs_below_the_floor_are_not_cut_on_either_side() {
        // The memo's row-granular rejection, enforced as a size: the same
        // declaration that cuts at MiB scale cuts nothing at KB scale, and it
        // crosses the floor at the same size in both packagings.
        let contract = contract(INTERLEAVED);
        let small = 12 * (MIB - 1024);
        assert!(
            contract
                .runs_of("b.0.qkv.weight", &[12, 2], small)
                .is_none()
        );
        assert!(
            contract
                .runs_of("b.0.q.weight", &[4, 2], small / 3)
                .is_none()
        );
        assert!(
            contract
                .seam_offsets("b.0.qkv.weight", &[12, 2], small)
                .is_empty()
        );
    }

    #[test]
    fn a_contract_is_addressed_by_name_and_version_and_pinned_by_digest() {
        let first = contract(FUSED);
        assert_eq!(first.stamp().to_string(), "test.fused@2");
        assert_eq!(Stamp::parse("test.fused@2").unwrap(), first.stamp());
        // Whitespace is not meaning: the same declarations digest the same.
        let reformatted = contract(&FUSED.replace("        ", " ").replace('\n', ""));
        assert_eq!(first.digest(), reformatted.digest());
        // A changed seam share is a changed contract.
        let edited = contract(&FUSED.replace("\"share\": 2", "\"share\": 3"));
        assert_ne!(first.digest(), edited.digest());
    }

    const NAMELESS: &str = r#"{
        "format": "tensorfs-contract-v1",
        "tensors": [
            {"role": "layers.{i}.attn.qkv", "pattern": "layers.{i}.qkv.weight", "rank": 2,
             "fusion": {"axis": 0, "parts": [{"role": "q", "share": 2}, {"role": "k", "share": 1},
                                             {"role": "v", "share": 1}]}},
            {"role": "layers.{i}.mlp", "pattern": "layers.{i}.mlp.weight", "required": false}
        ],
        "sets": {"adaln": ["layers.{i}.adaln.weight"]}
    }"#;

    #[test]
    fn a_nameless_contract_is_identified_by_its_digest_alone() {
        let custom = contract(NAMELESS);
        assert_eq!(custom.name(), None);
        assert_eq!(custom.version(), 0);

        // The stamp IS the canonical digest, spelled sha256:<hex>, and it
        // round-trips through parse/display like every other stamp.
        let stamp = custom.stamp();
        assert_eq!(stamp, Stamp::Digest(custom.digest()));
        let spelled = stamp.to_string();
        assert!(
            spelled.starts_with("sha256:") && spelled.len() == 71,
            "{spelled}"
        );
        assert_eq!(Stamp::parse(&spelled).unwrap(), stamp);

        // Whitespace is not meaning for customs either.
        let reformatted = contract(&NAMELESS.replace("        ", " ").replace('\n', ""));
        assert_eq!(custom.digest(), reformatted.digest());

        // The same declarations UNDER A NAME are a different contract: the
        // canonical rendering carries the name lines, so adoption into the
        // library is a new document with a new identity.
        assert_ne!(custom.digest(), contract(FUSED).digest());

        // A malformed digest spelling refuses, typed.
        assert!(Stamp::parse("sha256:abc").is_err());
        assert!(Stamp::parse(&spelled.to_uppercase()).is_err());
    }

    #[test]
    fn the_top_level_dtype_is_additive_and_read_back() {
        // Absent means absent: no field, no canonical line — the shipped
        // library's digest pins prove the byte-level half of this.
        assert_eq!(contract(FUSED).dtype(), None);

        let with = FUSED.replace(
            "\"version\": 2,",
            "\"version\": 2, \"dtype\": \"bfloat16\",",
        );
        let carried = contract(&with);
        assert_eq!(carried.dtype(), Some("bfloat16"));
        // Declaring it is meaning: the digest moves.
        assert_ne!(carried.digest(), contract(FUSED).digest());

        // The spelling is torch's, checked as a shape rather than an enum.
        let miscased = FUSED.replace("\"version\": 2,", "\"version\": 2, \"dtype\": \"BF16\",");
        assert!(matches!(
            Contract::parse(&miscased),
            Err(ContractError::Dtype(_))
        ));
        let empty = FUSED.replace("\"version\": 2,", "\"version\": 2, \"dtype\": \"\",");
        assert!(matches!(
            Contract::parse(&empty),
            Err(ContractError::Dtype(_))
        ));
    }

    #[test]
    fn name_and_version_are_declared_together_or_not_at_all() {
        let with_name_only =
            NAMELESS.replace("\"tensors\"", "\"name\": \"test.custom\", \"tensors\"");
        let with_version_only = NAMELESS.replace("\"tensors\"", "\"version\": 1, \"tensors\"");
        assert!(matches!(
            Contract::parse(&with_name_only),
            Err(ContractError::Identity)
        ));
        assert!(matches!(
            Contract::parse(&with_version_only),
            Err(ContractError::Identity)
        ));
    }

    #[test]
    fn malformed_documents_refuse() {
        let cases = [
            FUSED.replace(CONTRACT_FORMAT, "tensorfs-contract-v2"),
            FUSED.replace("\"version\": 2", "\"version\": 0"),
            FUSED.replace("test.fused", "Test.Fused"),
            // An inner-axis seam is not a byte concatenation.
            FUSED.replace("\"axis\": 0", "\"axis\": 1"),
            // One part in one group is the whole tensor, not a fusion.
            ONE_PART.to_owned(),
            // An unnamed part is only meaningful as the sole part of a slice.
            FUSED.replace(
                "\"role\": \"q\", \"share\": 2",
                "\"role\": \"\", \"share\": 2",
            ),
            FUSED.replace("\"axis\": 0", "\"axis\": 0, \"groups\": 0"),
            FUSED.replace(
                "\"role\": \"k\", \"share\": 1",
                "\"role\": \"k\", \"share\": 0",
            ),
            FUSED.replace("\"rank\": 2", "\"rank\": 2, \"unknown\": 1"),
        ];
        for case in cases {
            assert!(Contract::parse(&case).is_err(), "accepted {case}");
        }
        // Two parts with one role name cannot be told apart.
        assert!(Contract::parse(&FUSED.replace("\"role\": \"k\"", "\"role\": \"q\"")).is_err());
    }
}
