//! Layout-only adapters: serving contract A from a checkpoint written in
//! contract B, without storing a second copy.
//!
//! The consumer is serving. Endpoint code declares the layout it implements,
//! a checkpoint arrives spelled differently, and the question is mechanical:
//! **can this be served as a VIEW of bytes we already hold, or does it need a
//! real conversion?** [`decide`] answers it from the two contracts and the
//! file's header — same roles, same dtypes, same element counts.
//!
//! Three outcomes, and the middle one is the whole point:
//!
//! - [`Decision::RunPreserving`] — every target tensor is an ordered
//!   concatenation of source byte runs. Rename, reshape, and fuse/split along
//!   the outermost axis all land here. A derived snapshot is then an ORDINARY
//!   TFM1 manifest whose records point at the source's objects under new
//!   names and boundaries — no program, no new grammar, zero new data
//!   objects, and GC pins the sources through the existing mark walk. This is
//!   #80's re-key shape applied to a contract pair.
//! - [`Decision::Rearranged`] — same tensors, same bytes, different order
//!   inside some of them (the llama.cpp rope-permute). Byte-shareable it is
//!   not; viewable it is. Those tensors carry a recorded permute and are
//!   materialized once at definition time, which is what the load-order
//!   ruling asks for on a hot path: the stored copy is laid out in the order
//!   it will be read.
//! - [`Decision::Conversion`] — a tensor is missing, a dtype differs, or an
//!   element count differs. That is math, not layout, and it belongs to the
//!   conversion pipeline.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::contract::{Contract, FusionRun, Permute};
use crate::planner::{InventoryTensor, TensorInventory};

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("the source file does not implement the source contract")]
    SourceMismatch,
    #[error("{0}")]
    NotDerivable(String),
}

/// One contiguous byte run of the source file, identified by the role it
/// carries rather than by the tensor it happens to live in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRun {
    role: String,
    tensor: String,
    dtype: String,
    /// The re-arrangement the SOURCE layout applies to this role, if any.
    permute: Option<Permute>,
    /// Offset within the file.
    offset: u64,
    length: u64,
    /// Rows of the outermost axis this run covers, and the shape of one row.
    rows: u64,
    row_shape: Vec<u64>,
}

impl SourceRun {
    #[must_use]
    pub fn role(&self) -> &str {
        &self.role
    }

    #[must_use]
    pub fn tensor(&self) -> &str {
        &self.tensor
    }

    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    #[must_use]
    pub fn row_shape(&self) -> &[u64] {
        &self.row_shape
    }
}

/// One tensor of the derived layout: its name, its shape, and the ordered
/// source runs whose concatenation IS its bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    runs: Vec<SourceRun>,
    transform: Transform,
}

/// What must happen to the concatenated source runs to produce this tensor.
///
/// `Identity` is the whole point of the run-preserving class: the bytes are
/// already right, so the derived snapshot is an ordinary manifest. The other
/// two are a generalized permute in one direction or the other, applied once
/// at definition time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Transform {
    Identity,
    Forward(Permute),
    Inverse(Permute),
}

impl Transform {
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

impl DerivedTensor {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[must_use]
    pub fn runs(&self) -> &[SourceRun] {
        &self.runs
    }

    /// The re-arrangement needed inside this tensor after concatenation.
    #[must_use]
    pub const fn transform(&self) -> &Transform {
        &self.transform
    }

    #[must_use]
    pub fn length(&self) -> u64 {
        self.runs.iter().map(|run| run.length).sum()
    }

    /// True when this tensor already exists in the source under one name and
    /// one contiguous extent — a pure rename or reshape.
    #[must_use]
    pub fn is_whole_source_tensor(&self) -> bool {
        matches!(self.runs.as_slice(), [only] if only.rows == self.shape.first().copied().unwrap_or(0))
    }
}

/// A complete layout adapter: every tensor of the target layout, expressed as
/// runs of the source's bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Adapter {
    tensors: Vec<DerivedTensor>,
}

impl Adapter {
    #[must_use]
    pub fn tensors(&self) -> &[DerivedTensor] {
        &self.tensors
    }
}

/// The answer to "view or convert?".
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Decision {
    /// Every target tensor is an ordered concatenation of source runs; the
    /// derived snapshot is an ordinary manifest over the source's objects.
    RunPreserving(Adapter),
    /// Serving is possible, but at least one tensor's bytes are re-arranged
    /// inside itself and must be materialized once.
    Rearranged {
        adapter: Adapter,
        rearranged: Vec<String>,
    },
    /// Not a layout change.
    Conversion { reason: String },
}

impl Decision {
    /// Whether the target layout can be served from the source's bytes at
    /// all — the serve-time question, with no storage question attached.
    #[must_use]
    pub const fn is_viewable(&self) -> bool {
        !matches!(self, Self::Conversion { .. })
    }

    #[must_use]
    pub const fn adapter(&self) -> Option<&Adapter> {
        match self {
            Self::RunPreserving(adapter) | Self::Rearranged { adapter, .. } => Some(adapter),
            Self::Conversion { .. } => None,
        }
    }
}

/// Every byte run of a file, keyed by the role the source contract gives it.
///
/// A tensor with no declared fusion is one run under its bare role; a fused
/// tensor is one run per declared part, under `role#part@group`. That naming
/// is the entire correspondence mechanism: two contracts that describe the
/// same weights agree on these strings, whatever they call the tensors.
pub fn role_runs(
    contract: &Contract,
    inventory: &TensorInventory,
) -> Result<BTreeMap<String, SourceRun>, AdapterError> {
    let mut runs = BTreeMap::new();
    for tensor in inventory.tensors() {
        let Some(entry) = contract.entry_for(tensor.name()) else {
            continue;
        };
        let captures = entry
            .pattern()
            .captures(tensor.name())
            .ok_or(AdapterError::SourceMismatch)?;
        let base = entry
            .role()
            .instantiate(&captures)
            .ok_or(AdapterError::SourceMismatch)?;
        let declared = contract.runs_of(tensor.name(), tensor.shape(), tensor.length());
        let parts: Vec<FusionRun> =
            declared.unwrap_or_else(|| vec![FusionRun::whole(tensor.length())]);
        let mut offset = tensor.offset();
        for part in parts {
            let role = format!("{base}{}", part.role());
            let mut run = source_run(tensor, &role, offset, part.length())?;
            run.permute = entry.permute().cloned();
            if runs.insert(role.clone(), run).is_some() {
                return Err(AdapterError::NotDerivable(format!(
                    "two source tensors carry the role {role:?}"
                )));
            }
            offset += part.length();
        }
    }
    Ok(runs)
}

fn source_run(
    tensor: &InventoryTensor,
    role: &str,
    offset: u64,
    length: u64,
) -> Result<SourceRun, AdapterError> {
    let outer = tensor.shape().first().copied().unwrap_or(1).max(1);
    let row_bytes = tensor.length() / outer;
    if row_bytes == 0 || !length.is_multiple_of(row_bytes) {
        return Err(AdapterError::NotDerivable(format!(
            "run {role:?} is not a whole number of rows"
        )));
    }
    Ok(SourceRun {
        role: role.to_owned(),
        tensor: tensor.name().to_owned(),
        dtype: tensor.dtype().to_owned(),
        permute: None,
        offset,
        length,
        rows: length / row_bytes,
        row_shape: tensor.shape().iter().skip(1).copied().collect(),
    })
}

/// Answers "view or convert?" for one file and one target layout.
///
/// The inputs are exactly the two contracts and the source's HEADER — names,
/// shapes, dtypes. No tensor byte is read to decide.
pub fn decide(
    source: &Contract,
    inventory: &TensorInventory,
    target: &Contract,
) -> Result<Decision, AdapterError> {
    let available = role_runs(source, inventory)?;
    let mut tensors = Vec::new();
    let mut claimed: BTreeSet<&str> = BTreeSet::new();

    for entry in target.tensors() {
        let suffixes = entry_suffixes(entry);
        // Which hole values does this declaration cover? They are read back
        // out of the roles the source actually carries, so the target never
        // has to be told the model's depth.
        let mut captures: BTreeSet<Vec<u64>> = BTreeSet::new();
        for role in available.keys() {
            for suffix in &suffixes {
                if let Some(base) = role.strip_suffix(suffix.as_str())
                    && let Some(values) = entry.role().captures(base)
                {
                    captures.insert(values);
                }
            }
        }

        if captures.is_empty() && entry.is_required() {
            return Ok(Decision::Conversion {
                reason: format!(
                    "the target requires {:?}, which the source has no bytes for",
                    entry.role().as_str()
                ),
            });
        }
        for values in captures {
            let name = entry
                .pattern()
                .instantiate(&values)
                .ok_or(AdapterError::SourceMismatch)?;
            let base = entry
                .role()
                .instantiate(&values)
                .ok_or(AdapterError::SourceMismatch)?;
            let mut runs = Vec::with_capacity(suffixes.len());
            for suffix in &suffixes {
                let role = format!("{base}{suffix}");
                let Some(run) = available.get(&role) else {
                    return Ok(Decision::Conversion {
                        reason: format!("the source carries no bytes for the role {role:?}"),
                    });
                };
                claimed.insert(run.role.as_str());
                runs.push(run.clone());
            }
            match derived_tensor(name, runs, entry.permute()) {
                Ok(tensor) => tensors.push(tensor),
                Err(DerivationFailure::NotALayoutChange(reason)) => {
                    return Ok(Decision::Conversion { reason });
                }
                Err(DerivationFailure::Inexpressible(reason)) => {
                    return Err(AdapterError::NotDerivable(reason));
                }
            }
        }
    }

    if tensors.is_empty() {
        return Ok(Decision::Conversion {
            reason: "the target contract describes none of this file".to_owned(),
        });
    }
    // Every source role must be spoken for: a target that silently drops
    // weights is a different model, not a view of this one.
    if let Some(orphan) = available
        .keys()
        .find(|role| !claimed.contains(role.as_str()))
    {
        return Ok(Decision::Conversion {
            reason: format!("the target layout has no place for the role {orphan:?}"),
        });
    }

    let rearranged: Vec<String> = tensors
        .iter()
        .filter(|tensor| !tensor.transform.is_identity())
        .map(|tensor| tensor.name.clone())
        .collect();
    let adapter = Adapter { tensors };
    if rearranged.is_empty() {
        Ok(Decision::RunPreserving(adapter))
    } else {
        Ok(Decision::Rearranged {
            adapter,
            rearranged,
        })
    }
}

/// The ordered run-role suffixes one target declaration needs, in file order.
fn entry_suffixes(entry: &crate::contract::TensorPattern) -> Vec<String> {
    let Some(fusion) = entry.fusion() else {
        return vec![String::new()];
    };
    let mut suffixes = Vec::new();
    for group in 0..fusion.groups() {
        for part in fusion.parts() {
            let mut suffix = String::new();
            if !part.role().is_empty() {
                suffix.push('#');
                suffix.push_str(part.role());
            }
            if fusion.groups() > 1 {
                suffix.push('@');
                suffix.push_str(&group.to_string());
            }
            suffixes.push(suffix);
        }
    }
    suffixes
}

/// Why a target tensor could not be derived: a fact about the weights, or a
/// limit of this vocabulary. They are different answers and must not be
/// reported as one.
enum DerivationFailure {
    NotALayoutChange(String),
    Inexpressible(String),
}

fn derived_tensor(
    name: String,
    runs: Vec<SourceRun>,
    target_permute: Option<&Permute>,
) -> Result<DerivedTensor, DerivationFailure> {
    let first = runs.first().ok_or_else(|| {
        DerivationFailure::NotALayoutChange("a target tensor with no runs".to_owned())
    })?;
    let dtype = first.dtype.clone();
    let row_shape = first.row_shape.clone();
    let source_permute = first.permute.clone();
    for run in &runs {
        if run.dtype != dtype {
            return Err(DerivationFailure::NotALayoutChange(format!(
                "{name:?} would concatenate {} and {} — a cast is math, not layout",
                dtype, run.dtype
            )));
        }
        if run.row_shape != row_shape {
            return Err(DerivationFailure::NotALayoutChange(format!(
                "{name:?} would concatenate rows of different shapes"
            )));
        }
        if run.permute != source_permute {
            return Err(DerivationFailure::Inexpressible(format!(
                "{name:?} would concatenate runs re-arranged differently"
            )));
        }
    }
    // A permute is declared relative to the role's canonical order, so the
    // transform between two layouts is target ∘ source⁻¹.
    let transform = match (source_permute, target_permute) {
        (None, None) => Transform::Identity,
        (Some(source), Some(target)) if source == *target => Transform::Identity,
        (None, Some(target)) => Transform::Forward(target.clone()),
        (Some(source), None) => Transform::Inverse(source),
        (Some(_), Some(_)) => {
            return Err(DerivationFailure::Inexpressible(format!(
                "{name:?} is re-arranged on BOTH sides; composing two permutes is not in this vocabulary"
            )));
        }
    };
    let mut shape = vec![runs.iter().map(|run| run.rows).sum()];
    shape.extend_from_slice(&row_shape);
    Ok(DerivedTensor {
        name,
        dtype,
        shape,
        runs,
        transform,
    })
}

// ---------------------------------------------------------------------------
// The permute kernel
// ---------------------------------------------------------------------------

/// Applies a generalized permute to one tensor's bytes.
///
/// `dims` are the resolved view dimensions of the SOURCE side of the permute
/// and `axes[t]` names the source axis that becomes output axis `t`. The
/// operation is a pure index remap of fixed-size elements: no arithmetic
/// touches a value, so it is exactly invertible and dtype-blind — which is
/// what makes it layout rather than math.
///
/// When the innermost axis is unmoved (the rope-permute case), whole rows are
/// copied instead of elements.
#[must_use]
pub fn permute_bytes(input: &[u8], element_size: usize, dims: &[u64], axes: &[usize]) -> Vec<u8> {
    debug_assert_eq!(dims.len(), axes.len());
    let rank = dims.len();
    let dimensions: Vec<usize> = dims.iter().map(|value| *value as usize).collect();
    let elements: usize = dimensions.iter().product();
    debug_assert_eq!(elements * element_size, input.len());

    // Row-major strides of the source view.
    let mut strides = vec![1_usize; rank];
    for axis in (0..rank.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * dimensions[axis + 1];
    }
    let out_dims: Vec<usize> = axes.iter().map(|axis| dimensions[*axis]).collect();
    let out_strides: Vec<usize> = axes.iter().map(|axis| strides[*axis]).collect();

    // The innermost output axis is contiguous in the source exactly when it
    // is the innermost source axis; then one copy moves a whole row.
    let block = if rank > 0 && axes[rank - 1] == rank - 1 {
        out_dims[rank - 1]
    } else {
        1
    };
    let outer = rank - usize::from(block > 1);

    let mut output = vec![0_u8; input.len()];
    let mut counter = vec![0_usize; outer];
    let mut written = 0_usize;
    loop {
        let source: usize = counter
            .iter()
            .zip(&out_strides[..outer])
            .map(|(index, stride)| index * stride)
            .sum();
        let from = source * element_size;
        let span = block * element_size;
        output[written..written + span].copy_from_slice(&input[from..from + span]);
        written += span;
        if written == output.len() {
            break;
        }
        let mut axis = outer;
        loop {
            if axis == 0 {
                break;
            }
            axis -= 1;
            counter[axis] += 1;
            if counter[axis] < out_dims[axis] {
                break;
            }
            counter[axis] = 0;
        }
    }
    output
}

/// The permutation that undoes `axes`.
#[must_use]
pub fn invert_axes(axes: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; axes.len()];
    for (position, axis) in axes.iter().enumerate() {
        inverse[*axis] = position;
    }
    inverse
}

impl Transform {
    /// Applies this transform to one tensor's concatenated bytes.
    ///
    /// `shape` is the tensor's logical shape on the side the bytes are
    /// currently in, which is what the declaration resolves against.
    #[must_use]
    pub fn apply(&self, bytes: &[u8], element_size: usize, shape: &[u64]) -> Option<Vec<u8>> {
        match self {
            Self::Identity => Some(bytes.to_vec()),
            Self::Forward(permute) => {
                let dims = permute.resolve(shape)?;
                Some(permute_bytes(bytes, element_size, &dims, permute.axes()))
            }
            Self::Inverse(permute) => {
                // The stored bytes are already permuted, so the view they sit
                // in is the PERMUTED one; undoing it means walking back.
                let dims = permute.resolve(shape)?;
                let permuted: Vec<u64> = permute.axes().iter().map(|axis| dims[*axis]).collect();
                Some(permute_bytes(
                    bytes,
                    element_size,
                    &permuted,
                    &invert_axes(permute.axes()),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy: `arange(24).reshape(2, 3, 4).transpose(1, 0, 2)`.
    #[test]
    fn a_permute_moves_elements_exactly_where_numpy_does() {
        let input: Vec<u8> = (0..24).collect();
        let moved = permute_bytes(&input, 1, &[2, 3, 4], &[1, 0, 2]);
        assert_eq!(
            moved,
            vec![
                0, 1, 2, 3, 12, 13, 14, 15, // (0,0,:), (1,0,:)
                4, 5, 6, 7, 16, 17, 18, 19, // (0,1,:), (1,1,:)
                8, 9, 10, 11, 20, 21, 22, 23,
            ]
        );
    }

    /// The llama.cpp rope permute: `reshape(heads, 2, d/2, cols).swapaxes(1, 2)`.
    #[test]
    fn the_rope_permute_round_trips_exactly() {
        let dims = [2_u64, 2, 3, 2];
        let axes = [0_usize, 2, 1, 3];
        let elements = 2 * 2 * 3 * 2;
        let input: Vec<u8> = (0..elements as u8 * 2).collect();
        let moved = permute_bytes(&input, 2, &dims, &axes);
        assert_ne!(moved, input, "a rope permute is not the identity");

        let permuted: Vec<u64> = axes.iter().map(|axis| dims[*axis]).collect();
        let back = permute_bytes(&moved, 2, &permuted, &invert_axes(&axes));
        assert_eq!(back, input, "a permute is exactly invertible");
    }

    #[test]
    fn an_inner_axis_move_still_lands_element_by_element() {
        // axes[last] != last, so the block-copy shortcut is off.
        let input: Vec<u8> = (0..6).collect();
        assert_eq!(
            permute_bytes(&input, 1, &[2, 3], &[1, 0]),
            vec![0, 3, 1, 4, 2, 5]
        );
    }
}
