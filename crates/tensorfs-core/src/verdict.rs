//! The bind verdict — `Satisfies | Derivable | Incompatible` — in Rust.
//!
//! This is `verdict.go` + `match.go` ported rule for rule, and the port exists
//! because the verdict had exactly ONE implementation (Go) while the caller
//! that most needs it — gen-worker's boot-time lane selection — is Python. A
//! narrow re-implementation on that side would be the THIRD copy of the pattern
//! matcher (`python/src/tensorfs/convert.py` is the second, and tensorfs#129
//! already names the duplication), and three copies of a rule that decides
//! whether a checkpoint may load is three chances to admit a bind that should
//! have been refused. A wrong ADMIT is invisible until a pod 500s.
//!
//! Semantics are GO-IDENTICAL, including the rendered strings, so parity is
//! provable by running both over the same fixtures rather than by reading both.
//! Where Go deliberately diverges from `Contract::matches`, that divergence is
//! reproduced here rather than "fixed":
//!
//! * `Length == 0` means NOT SUPPLIED, and the byte half of the fusion rule is
//!   skipped while the shape half still applies. A gate that invented a refusal
//!   out of an absent number would refuse checkpoints the pipeline runs.
//! * the 1 MiB seam floor is NOT applied here. `Fusion::runs` applies it because
//!   it is deciding cut points; matching only asks whether the declaration can
//!   apply at all. A fusion below the floor matches and yields no seams.

use std::collections::BTreeSet;
use std::fmt;

use crate::contract::{Contract, Fusion, FusionPart, Permute, TensorPattern};

/// One header entry, reduced to what a contract check needs.
///
/// `dtype` is the container's own spelling (`"BF16"`, `"F16"`, a ggml type
/// name), because that is what a contract's `dtypes` lists. `shape` is logical
/// row-major, outermost axis first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte extent, read ONLY by the fusion divisibility check. Zero means
    /// "not supplied" and is PERMISSIVE — see the module note.
    pub length: u64,
}

/// One tensor-carrying member of a checkpoint. Matching is PER FILE: a
/// multifolder lane document is all-optional declarations, and each component
/// file matches the families it carries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFile {
    pub path: String,
    pub tensors: Vec<InventoryTensor>,
}

/// A successful contract match over one file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileMatch {
    pub stamp: String,
    /// TENSORS EXPLAINED, not declarations satisfied: several tensors hit one
    /// `{i}` declaration and each is counted.
    pub matched: usize,
}

/// Why a contract did not describe a file. Stable labels, meant to reach a user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MismatchKind {
    /// The one kind that is routinely a PACKAGING difference rather than a
    /// different model — which is what makes the middle verdict mandatory.
    Dtype,
    Rank,
    Fusion,
    Permute,
    Required,
    /// The floor: the contract claimed no tensor of this file at all. Without
    /// it every all-optional document would vacuously match every tensor
    /// container in existence.
    NothingExplained,
}

impl MismatchKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dtype => "dtype",
            Self::Rank => "rank",
            Self::Fusion => "fusion",
            Self::Permute => "permute",
            Self::Required => "required",
            Self::NothingExplained => "nothing-explained",
        }
    }
}

/// A NAMED refusal. `tensor` and `pattern` are the whole point: a verdict an
/// operator cannot act on is not a verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mismatch {
    pub kind: MismatchKind,
    pub tensor: String,
    pub pattern: String,
    pub role: String,
    pub declared: String,
    pub observed: String,
    pub stamp: String,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            MismatchKind::NothingExplained => {
                write!(out, "{} explains no tensor of this file", self.stamp)
            }
            MismatchKind::Required => write!(
                out,
                "{} requires {:?}, which no tensor satisfies",
                self.stamp, self.pattern
            ),
            kind => write!(
                out,
                "{}: tensor {:?} matches pattern {:?} but its {} is {}, not {}",
                self.stamp,
                self.tensor,
                self.pattern,
                kind.as_str(),
                self.observed,
                self.declared
            ),
        }
    }
}

/// The three-way answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerdictKind {
    Satisfies,
    /// The gap is one a NAMED conversion closes. The caller's obligation is to
    /// OFFER that conversion, not to refuse.
    Derivable,
    Incompatible,
}

impl VerdictKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Satisfies => "satisfies",
            Self::Derivable => "derivable",
            Self::Incompatible => "incompatible",
        }
    }
}

/// `dtype-cast` changes element types and nothing else.
pub const RECIPE_DTYPE_CAST: &str = "dtype-cast";
/// `fp8-rowwise` quantizes to fp8 and EMITS a per-row scale per weight.
pub const RECIPE_FP8_ROWWISE: &str = "fp8-rowwise";

/// The work that would make a Derivable artifact satisfy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Conversion {
    /// The RECIPE, not the observation: it comes from the TARGET contract,
    /// because the remedy for "these bytes are the wrong element type" depends
    /// on what the target lane's bytes ARE.
    pub kind: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub files: Vec<String>,
}

impl fmt::Display for Conversion {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "{} from {} to {} ({} file(s): {})",
            self.kind,
            self.from.join("|"),
            self.to.join("|"),
            self.files.len(),
            self.files.join(", ")
        )
    }
}

/// The artifact-level answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Verdict {
    pub kind: VerdictKind,
    pub stamp: String,
    pub matched: usize,
    pub explained: usize,
    /// Members the contract claimed no tensor of. NOT a refusal on its own.
    pub unexplained: Vec<String>,
    /// Set iff `kind` is `Derivable`.
    pub conversion: Option<Conversion>,
    /// Set iff `kind` is `Incompatible` or `Derivable`.
    pub mismatch: Option<Mismatch>,
    /// The MEMBER the mismatch came from. Load-bearing: the same contract
    /// routinely matches some members and refuses others, and a refusal that
    /// names only the tensor leaves the operator guessing which component.
    pub file: String,
}

impl fmt::Display for Verdict {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            VerdictKind::Satisfies => write!(
                out,
                "satisfies {} ({} tensors across {} file(s))",
                self.stamp, self.matched, self.explained
            ),
            VerdictKind::Derivable => write!(
                out,
                "derivable to {} via {} — in {}, {}",
                self.stamp,
                self.conversion
                    .as_ref()
                    .expect("derivable carries a conversion"),
                self.file,
                self.mismatch
                    .as_ref()
                    .expect("derivable carries a mismatch")
            ),
            VerdictKind::Incompatible => {
                let mismatch = self
                    .mismatch
                    .as_ref()
                    .expect("incompatible carries a mismatch");
                if self.file.is_empty() {
                    write!(out, "incompatible with {} — {}", self.stamp, mismatch)
                } else {
                    write!(
                        out,
                        "incompatible with {} — in {}, {}",
                        self.stamp, self.file, mismatch
                    )
                }
            }
        }
    }
}

impl Contract {
    /// The conversion this contract's bytes are MADE by, derived from the
    /// declarations and stored nowhere (tcg#53). Note what it does not read:
    /// the name. A document called `…-fp8-rowwise` that failed to declare
    /// scales answers `dtype-cast`, because a name is a label and the
    /// declarations are the falsifiable part.
    #[must_use]
    pub fn recipe(&self) -> &'static str {
        let mut fp8 = false;
        let mut scales = false;
        for decl in self.tensors() {
            if decl
                .dtypes()
                .iter()
                .any(|dtype| dtype == "F8_E4M3" || dtype == "F8_E5M2")
            {
                fp8 = true;
            }
            if decl.pattern().as_str().ends_with(".weight_scale") {
                scales = true;
            }
        }
        if fp8 && scales {
            RECIPE_FP8_ROWWISE
        } else {
            RECIPE_DTYPE_CAST
        }
    }

    /// Whether this contract describes one file, naming the FIRST disagreement
    /// when it does not.
    ///
    /// This is `Contract::matches` with the refusal reason kept instead of
    /// collapsed to `None`: the yes/no callers only ask yes/no, and the whole
    /// value of a bind-time verdict is the reason.
    ///
    /// # Errors
    /// The named mismatch, when the contract does not describe the file.
    pub fn match_file(&self, inventory: &[InventoryTensor]) -> Result<FileMatch, Box<Mismatch>> {
        let stamp = self.stamp().to_string();
        let declarations = self.tensors();
        let mut seen = vec![false; declarations.len()];
        let mut matched = 0_usize;

        for tensor in inventory {
            // FIRST declaration wins, in document order — the same rule as
            // `matches`. Declaration order is part of a contract's meaning.
            let Some(index) = declarations
                .iter()
                .position(|decl| decl.pattern().matches(&tensor.name))
            else {
                // An unclaimed tensor is neither a refusal nor a credit: a file
                // may carry arbitrary extra tensors, which is precisely why a
                // 9-channel inpainting `conv_in` the SDXL contract never claims
                // gets through (tensorfs#122, an ACCEPTED admit).
                continue;
            };
            if let Some(bad) = accepts(&declarations[index], &stamp, tensor) {
                return Err(Box::new(bad));
            }
            seen[index] = true;
            matched += 1;
        }

        for (index, decl) in declarations.iter().enumerate() {
            if decl.is_required() && !seen[index] {
                return Err(Box::new(Mismatch {
                    kind: MismatchKind::Required,
                    tensor: String::new(),
                    pattern: decl.pattern().as_str().to_owned(),
                    role: decl.role().as_str().to_owned(),
                    declared: String::new(),
                    observed: String::new(),
                    stamp,
                }));
            }
        }
        if matched == 0 {
            return Err(Box::new(Mismatch {
                kind: MismatchKind::NothingExplained,
                tensor: String::new(),
                pattern: String::new(),
                role: String::new(),
                declared: String::new(),
                observed: String::new(),
                stamp,
            }));
        }
        Ok(FileMatch { stamp, matched })
    }

    /// The bind question for a whole artifact, applied in this order:
    ///
    /// 1. a STRUCTURAL disagreement is incompatible outright — no cast makes a
    ///    differently-shaped tensor the right tensor, and it is reported first
    ///    so a conversion is never offered for work that would not help;
    /// 2. otherwise a DTYPE-only disagreement makes the artifact derivable;
    /// 3. otherwise, if at least one member was explained, it satisfies;
    /// 4. otherwise nothing here is this contract's: incompatible.
    ///
    /// Members are visited in path order, so the tensor a verdict names is
    /// stable across calls; a refusal that names a different tensor each time
    /// reads as flakiness.
    #[must_use]
    pub fn verdict(&self, files: &[ArtifactFile]) -> Verdict {
        let mut ordered: Vec<&ArtifactFile> = files.iter().collect();
        ordered.sort_by(|left, right| left.path.cmp(&right.path));

        let stamp = self.stamp().to_string();
        let mut matched = 0_usize;
        let mut explained = 0_usize;
        let mut unexplained: Vec<String> = Vec::new();
        let mut structural: Option<(Mismatch, String)> = None;
        let mut dtype_first: Option<(Mismatch, String)> = None;
        let mut from: BTreeSet<String> = BTreeSet::new();
        let mut to: BTreeSet<String> = BTreeSet::new();
        let mut cast_files: Vec<String> = Vec::new();

        for file in ordered {
            match self.match_file(&file.tensors) {
                Ok(found) => {
                    matched += found.matched;
                    explained += 1;
                }
                Err(mismatch) => match mismatch.kind {
                    MismatchKind::NothingExplained => unexplained.push(file.path.clone()),
                    MismatchKind::Dtype => {
                        from.insert(mismatch.observed.clone());
                        for declared in mismatch.declared.split('|') {
                            to.insert(declared.to_owned());
                        }
                        cast_files.push(file.path.clone());
                        if dtype_first.is_none() {
                            dtype_first = Some((*mismatch, file.path.clone()));
                        }
                    }
                    _ => {
                        if structural.is_none() {
                            structural = Some((*mismatch, file.path.clone()));
                        }
                    }
                },
            }
        }

        if let Some((mismatch, file)) = structural {
            return Verdict {
                kind: VerdictKind::Incompatible,
                stamp,
                matched,
                explained,
                unexplained,
                conversion: None,
                mismatch: Some(mismatch),
                file,
            };
        }
        if let Some((mismatch, file)) = dtype_first {
            let conversion = Conversion {
                kind: self.recipe().to_owned(),
                from: from.into_iter().collect(),
                to: to.into_iter().collect(),
                files: cast_files,
            };
            return Verdict {
                kind: VerdictKind::Derivable,
                stamp,
                matched,
                explained,
                unexplained,
                conversion: Some(conversion),
                mismatch: Some(mismatch),
                file,
            };
        }
        if explained > 0 {
            return Verdict {
                kind: VerdictKind::Satisfies,
                stamp,
                matched,
                explained,
                unexplained,
                conversion: None,
                mismatch: None,
                file: String::new(),
            };
        }
        let mismatch = Mismatch {
            kind: MismatchKind::NothingExplained,
            tensor: String::new(),
            pattern: String::new(),
            role: String::new(),
            declared: String::new(),
            observed: String::new(),
            stamp: stamp.clone(),
        };
        Verdict {
            kind: VerdictKind::Incompatible,
            stamp,
            matched,
            explained,
            unexplained,
            conversion: None,
            mismatch: Some(mismatch),
            file: String::new(),
        }
    }
}

/// The per-tensor half: dtype, rank, then the seam arithmetic. A claimed tensor
/// that disagrees is a refusal, not a shrug — contracts must be falsifiable
/// from the header alone or a stamp would mean nothing.
fn accepts(decl: &TensorPattern, stamp: &str, tensor: &InventoryTensor) -> Option<Mismatch> {
    let bad = |kind: MismatchKind, declared: String, observed: String| Mismatch {
        kind,
        tensor: tensor.name.clone(),
        pattern: decl.pattern().as_str().to_owned(),
        role: decl.role().as_str().to_owned(),
        declared,
        observed,
        stamp: stamp.to_owned(),
    };
    let dtypes = decl.dtypes();
    if !dtypes.is_empty() && !dtypes.iter().any(|declared| declared == &tensor.dtype) {
        return Some(bad(
            MismatchKind::Dtype,
            dtypes.join("|"),
            tensor.dtype.clone(),
        ));
    }
    if let Some(rank) = decl.rank()
        && rank != tensor.shape.len()
    {
        return Some(bad(
            MismatchKind::Rank,
            rank.to_string(),
            tensor.shape.len().to_string(),
        ));
    }
    if let Some(fusion) = decl.fusion()
        && !divides(fusion, &tensor.shape, tensor.length)
    {
        return Some(bad(
            MismatchKind::Fusion,
            describe(fusion),
            shape_text(&tensor.shape),
        ));
    }
    if let Some(permute) = decl.permute()
        && !resolves(permute, &tensor.shape)
    {
        return Some(bad(
            MismatchKind::Permute,
            "a resolvable view".to_owned(),
            shape_text(&tensor.shape),
        ));
    }
    None
}

/// `Fusion::runs` reduced to its yes/no, WITHOUT the seam floor — matching asks
/// whether the declaration can apply, not where the cuts land.
fn divides(fusion: &Fusion, shape: &[u64], length: u64) -> bool {
    let Some(&rows) = shape.first() else {
        return false;
    };
    let per_group: u64 = fusion.parts().iter().map(FusionPart::share).sum();
    let Some(units) = per_group.checked_mul(fusion.groups()) else {
        return false;
    };
    if rows == 0 || units == 0 || !rows.is_multiple_of(units) {
        return false;
    }
    // Length 0 is "not supplied": skipping the byte half is the permissive
    // divergence, never a manufactured refusal.
    length == 0 || length.is_multiple_of(rows)
}

fn describe(fusion: &Fusion) -> String {
    let parts: Vec<String> = fusion
        .parts()
        .iter()
        .map(|part| {
            let role = if part.role().is_empty() {
                "-"
            } else {
                part.role()
            };
            format!("{role}:{}", part.share())
        })
        .collect();
    format!(
        "an outer axis divisible by {} group(s) of [{}]",
        fusion.groups(),
        parts.join(" ")
    )
}

/// `Permute::resolve` reduced to its yes/no.
fn resolves(permute: &Permute, shape: &[u64]) -> bool {
    permute.resolve(shape).is_some()
}

fn shape_text(shape: &[u64]) -> String {
    let parts: Vec<String> = shape.iter().map(u64::to_string).collect();
    format!("[{}]", parts.join(","))
}
