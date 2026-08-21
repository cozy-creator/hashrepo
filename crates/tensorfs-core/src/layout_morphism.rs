//! LAYOUT MORPHISMS: the ONE implementation that moves the bytes.
//!
//! A layout morphism is data — a factorization of a tensor's logical shape into
//! sub-axes and a permutation of those sub-axes — authored once in
//! `spec/v2/layouts/*.json` and vendored into this crate at build time. The Go
//! decision engine parses the same files to decide WHETHER a rearrangement is
//! allowed and what it costs. This module is the other half: given a record and
//! a shape, which byte goes where, and the copy that puts it there.
//!
//! There is exactly one applier and it runs in both directions. The inverse is
//! not a second map read backwards by a second piece of code; it is the same
//! walk with source and destination swapped. That is what makes
//! AUTO-RATIFICATION mean something: apply the map, apply it inverted, compare
//! bytes. A record that survives is ratified by arithmetic instead of by
//! somebody reviewing a permutation — which is the review that missed
//! nunchaku's `(4, 4, 8)` row split reshaping without error into `(4, 8, 4)`.
//!
//! Two backends, one implementation. `materialize` is CPU bytes to CPU bytes,
//! for a derived tree the storage tier decided to write down. The GPU fill
//! (`crate::layout_fill`, feature `cuda`) applies the SAME walk into pinned
//! staging and hands the result to the caller's device pointer — tensorfs never
//! allocates device memory; varena owns the address space.
//!
//! PADDING IS PART OF THE LANGUAGE. A sub-axis product that exceeds its logical
//! dim rounds up: the destination is bigger than the source, the fill elements
//! are zero, and the map is injective rather than bijective. Such a record
//! round-trips ON ITS IMAGE, which the verifier checks by name.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

include!(concat!(env!("OUT_DIR"), "/layout_catalog.rs"));

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("layout catalog {file}: {detail}")]
    Catalog { file: String, detail: String },
    #[error("{handle}: {detail}")]
    Record { handle: String, detail: String },
    #[error("{handle} arranges rank-{want} tensors; this one is rank {got}")]
    Rank {
        handle: String,
        want: usize,
        got: usize,
    },
    #[error("{handle}: axis {axis} factors into {product} elements and the axis is {dim}")]
    ShortAxis {
        handle: String,
        axis: usize,
        product: u64,
        dim: u64,
    },
    #[error("{handle}: {detail}")]
    Formula { handle: String, detail: String },
    #[error("buffer holds {got} bytes, the plan needs {want}")]
    Buffer { got: usize, want: usize },
    #[error("{handle} is an unratified candidate and does not derive")]
    Candidate { handle: String },
}

/// The two classes, and they are worth different money: the compiler's own
/// closed vocabulary, and the arrangement a kernel library demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutClass {
    Inductor,
    EndpointDeclared,
}

/// One factor of one logical axis.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubAxis {
    pub axis: usize,
    pub extent: String,
}

/// One validated storage arrangement, as read from the catalog.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutMorphism {
    pub format: String,
    pub name: String,
    pub version: u32,
    pub class: LayoutClass,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub rank: usize,
    #[serde(default)]
    pub sub_axes: Vec<SubAxis>,
    #[serde(default)]
    pub permutation: Vec<usize>,
    /// A wish a compiler emitted that nothing has ratified. It parses, it
    /// verifies, and it still never derives.
    #[serde(default)]
    pub candidate: bool,
    #[serde(default)]
    pub provenance: Vec<String>,
    #[serde(default)]
    pub digest: String,
}

/// The document format tag. A reader that guessed from shape would accept a
/// quant rule as a layout.
pub const LAYOUT_MORPHISM_FORMAT: &str = "tensorfs-layout-morphism-v2";

/// The arrangement every tree carries unless it says otherwise.
pub const DEFAULT_LAYOUT: &str = "torch.contiguous@1";

impl LayoutMorphism {
    /// `<name>@<version>`, the spelling a stamp and a bridge id use.
    pub fn handle(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }

    /// The identity: no factorization, no permutation, every rank.
    pub fn is_identity(&self) -> bool {
        self.sub_axes.is_empty()
    }

    /// Refuse an unratified candidate before it moves a byte.
    ///
    /// A candidate is a WISH a compiler emitted. It parses, it auto-verifies,
    /// and it still does not derive: machines derive along ratified morphisms,
    /// they never invent one. The gate is here as well as in the Go engine
    /// because this crate is what a worker actually calls.
    pub fn ensure_ratified(&self) -> Result<(), LayoutError> {
        if self.candidate {
            return Err(LayoutError::Candidate {
                handle: self.handle(),
            });
        }
        Ok(())
    }

    fn check(&self) -> Result<(), LayoutError> {
        let handle = self.handle();
        let bad = |detail: String| LayoutError::Record {
            handle: handle.clone(),
            detail,
        };
        if self.format != LAYOUT_MORPHISM_FORMAT {
            return Err(bad(format!("not a {LAYOUT_MORPHISM_FORMAT} document")));
        }
        if self.provenance.is_empty() {
            return Err(bad("carries no evidence".into()));
        }
        if self.is_identity() {
            if !self.permutation.is_empty() {
                return Err(bad("permutes a factorization it does not declare".into()));
            }
            if self.rank != 0 {
                return Err(bad("the identity is the identity at every rank".into()));
            }
            return Ok(());
        }
        if self.rank == 0 {
            return Err(bad("factors sub-axes without declaring a rank".into()));
        }
        if self.permutation.len() != self.sub_axes.len() {
            return Err(bad(format!(
                "permutes {} positions over {} sub-axes",
                self.permutation.len(),
                self.sub_axes.len()
            )));
        }
        let mut axis = 0usize;
        let mut covered = 0usize;
        for (at, sub) in self.sub_axes.iter().enumerate() {
            if at == 0 {
                if sub.axis != 0 {
                    return Err(bad("does not factor axis 0".into()));
                }
                covered = 1;
            } else if sub.axis == axis + 1 {
                covered += 1;
            } else if sub.axis != axis {
                return Err(bad(format!(
                    "lists axis {} after axis {axis}; sub-axes are grouped by \
                     axis, outermost first, and no axis is skipped",
                    sub.axis
                )));
            }
            axis = sub.axis;
        }
        if covered != self.rank {
            return Err(bad(format!(
                "declares rank {} and factors {covered} axes",
                self.rank
            )));
        }
        let mut seen = vec![false; self.permutation.len()];
        for &to in &self.permutation {
            let slot = seen.get_mut(to).ok_or_else(|| {
                bad(format!("permutes to position {to}, which does not exist"))
            })?;
            if *slot {
                return Err(bad(format!(
                    "sends two sub-axes to storage position {to}. A layout that \
                     is not a bijection loses elements, and it loses them silently"
                )));
            }
            *slot = true;
        }
        Ok(())
    }

    /// Plan this arrangement for one concrete logical shape.
    pub fn plan(&self, shape: &[u64]) -> Result<Plan, LayoutError> {
        let handle = self.handle();
        if self.rank != 0 && shape.len() != self.rank {
            return Err(LayoutError::Rank {
                handle,
                want: self.rank,
                got: shape.len(),
            });
        }
        if shape.is_empty() || shape.contains(&0) {
            return Err(LayoutError::Record {
                handle,
                detail: format!("cannot arrange a tensor of shape {shape:?}"),
            });
        }
        let mut source_elements: u64 = 1;
        for &dim in shape {
            source_elements = source_elements.saturating_mul(dim);
        }
        let mut logical_stride = vec![1u64; shape.len()];
        for at in (0..shape.len().saturating_sub(1)).rev() {
            logical_stride[at] = logical_stride[at + 1] * shape[at + 1];
        }

        // The identity needs no factorization: one sub-axis per logical axis,
        // in order. Written out rather than special-cased downstream, so the
        // walker has exactly one shape of input.
        let (axes, extents): (Vec<usize>, Vec<u64>) = if self.is_identity() {
            ((0..shape.len()).collect(), shape.to_vec())
        } else {
            let mut axes = Vec::with_capacity(self.sub_axes.len());
            let mut extents = Vec::with_capacity(self.sub_axes.len());
            for sub in &self.sub_axes {
                let extent = eval(&sub.extent, shape).map_err(|detail| LayoutError::Formula {
                    handle: handle.clone(),
                    detail: format!("axis {}: {detail}", sub.axis),
                })?;
                if extent == 0 {
                    return Err(LayoutError::Formula {
                        handle: handle.clone(),
                        detail: format!("axis {}: {:?} is zero-sized", sub.axis, sub.extent),
                    });
                }
                axes.push(sub.axis);
                extents.push(extent);
            }
            (axes, extents)
        };

        // `within` is the multiplier of a sub-axis coordinate inside its logical
        // axis: the product of the factors that follow it in the same axis.
        let mut within = vec![1u64; extents.len()];
        let mut padded_axis = vec![false; shape.len()];
        let mut padded = false;
        let mut at = 0usize;
        while at < axes.len() {
            let mut end = at;
            while end < axes.len() && axes[end] == axes[at] {
                end += 1;
            }
            let mut running = 1u64;
            for index in (at..end).rev() {
                within[index] = running;
                running *= extents[index];
            }
            let axis = axes[at];
            if axis >= shape.len() {
                return Err(LayoutError::Rank {
                    handle,
                    want: self.rank,
                    got: shape.len(),
                });
            }
            if running < shape[axis] {
                return Err(LayoutError::ShortAxis {
                    handle,
                    axis,
                    product: running,
                    dim: shape[axis],
                });
            }
            if running > shape[axis] {
                padded = true;
                padded_axis[axis] = true;
            }
            at = end;
        }

        let permutation = if self.is_identity() {
            (0..extents.len()).collect()
        } else {
            self.permutation.clone()
        };
        let src_stride: Vec<u64> = (0..extents.len())
            .map(|at| within[at] * logical_stride[axes[at]])
            .collect();
        let dest_elements = extents.iter().product();
        Ok(Plan {
            handle,
            shape: shape.to_vec(),
            axes,
            extents,
            within,
            permutation,
            logical_stride,
            src_stride,
            padded_axis,
            source_elements,
            dest_elements,
            padded,
        })
    }
}

/// What a layout morphism MEANS for one tensor: every number the walk needs.
#[derive(Debug, Clone)]
pub struct Plan {
    handle: String,
    shape: Vec<u64>,
    /// The logical axis each sub-axis belongs to, in declaration order.
    axes: Vec<usize>,
    /// Sub-axis extents, in declaration order.
    extents: Vec<u64>,
    /// The multiplier of each sub-axis coordinate inside its logical axis.
    within: Vec<u64>,
    /// `permutation[p]` is the declaration index at storage position `p`.
    permutation: Vec<usize>,
    logical_stride: Vec<u64>,
    /// The source-linear stride of each sub-axis coordinate, in declaration
    /// order: `within[j] * logical_stride[axes[j]]`. This is what decides
    /// whether a run of sub-axes is CONTIGUOUS in the source.
    src_stride: Vec<u64>,
    /// Per logical axis: does its factorization overshoot the dim? A padded
    /// axis cannot be folded into a contiguous run, because the run would span
    /// elements that have no source.
    padded_axis: Vec<bool>,
    source_elements: u64,
    dest_elements: u64,
    padded: bool,
}

impl Plan {
    pub fn handle(&self) -> &str {
        &self.handle
    }
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
    pub fn source_elements(&self) -> u64 {
        self.source_elements
    }
    pub fn dest_elements(&self) -> u64 {
        self.dest_elements
    }
    /// True when the destination is LARGER than the source because some axis
    /// rounded up. The map is then injective, not bijective.
    pub fn padded(&self) -> bool {
        self.padded
    }
    /// Sub-axis extents in STORAGE order — the destination's own shape.
    pub fn storage_extents(&self) -> Vec<u64> {
        self.permutation.iter().map(|&at| self.extents[at]).collect()
    }

    /// THE WALK, in maximal contiguous RUNS: `(destination element, source
    /// element or padding, run length)`, visiting every destination element
    /// exactly once across all runs, in destination order.
    ///
    /// This is the primitive, and it is computed ANALYTICALLY rather than by
    /// visiting elements and noticing that some were adjacent. The difference
    /// is not a micro-optimisation: a plan that is one run has to COST one run.
    /// Measured on a card, an element-by-element walk moved 16 MiB in 765 ms —
    /// 0.02 GiB/s for what is a single `memcpy` plus a single DMA — because it
    /// stepped an odometer four million times to discover a fact the strides
    /// already stated.
    ///
    /// The fact they state: fold trailing STORAGE positions into one run for as
    /// long as each sub-axis's source stride equals the run length accumulated
    /// so far. Then only the outer positions need an odometer.
    ///
    ///  * `torch.contiguous@1` folds every position — one run, no odometer, a
    ///    whole tensor as one DMA at any rank or shape.
    ///  * `cublas.blockscale-128x4@1` folds its innermost 4 — runs of four
    ///    elements.
    ///  * `torch.channels_last-2d@1` folds nothing: its innermost storage axis
    ///    is the channel, whose source stride is H*W. One run per element, and
    ///    that is the arrangement's own shape, not an artefact.
    ///
    /// A PADDED axis is never folded. Padding has no source element, so a run
    /// spanning it would have to be part copy and part zero; keeping padded
    /// axes in the odometer makes every run wholly one or wholly the other.
    ///
    /// The decomposition is STRUCTURAL. Two runs it emits may still happen to
    /// be adjacent in both buffers — at an image boundary, `channels_last`'s
    /// last element really does sit next to the next image's first — and this
    /// does not merge them. That is deliberate: the structural count is a
    /// property of the arrangement, and a count that drifted with the shape's
    /// accidental adjacencies would be a number nobody could re-derive.
    pub fn for_each_run(&self, mut visit: impl FnMut(u64, Option<u64>, u64)) {
        let n = self.extents.len();
        let mut run = 1u64;
        let mut outer = n;
        for position in (0..n).rev() {
            let at = self.permutation[position];
            if self.padded_axis[self.axes[at]] || self.src_stride[at] != run {
                break;
            }
            run *= self.extents[at];
            outer = position;
        }

        let mut counters = vec![0u64; n];
        let mut coord = vec![0u64; self.shape.len()];
        let runs = self.dest_elements / run.max(1);
        for step in 0..runs {
            let mut source = Some(0u64);
            for (axis, &position) in coord.iter().enumerate() {
                if position >= self.shape[axis] {
                    source = None;
                    break;
                }
                if let Some(offset) = source.as_mut() {
                    *offset += position * self.logical_stride[axis];
                }
            }
            visit(step * run, source, run);

            // Odometer over the OUTER storage positions only.
            for position in (0..outer).rev() {
                let at = self.permutation[position];
                counters[at] += 1;
                coord[self.axes[at]] += self.within[at];
                if counters[at] < self.extents[at] {
                    break;
                }
                coord[self.axes[at]] -= counters[at] * self.within[at];
                counters[at] = 0;
            }
        }
    }

    /// The walk one element at a time. Expands [`Plan::for_each_run`] rather
    /// than re-deriving anything: the verifier wants per-element granularity,
    /// and every mover wants runs.
    pub fn for_each(&self, mut visit: impl FnMut(u64, Option<u64>)) {
        self.for_each_run(|to, from, length| {
            for step in 0..length {
                visit(to + step, from.map(|index| index + step));
            }
        });
    }

    /// How many runs this plan decomposes into. Measured, never guessed.
    pub fn run_count(&self) -> u64 {
        let mut runs = 0;
        self.for_each_run(|_, _, _| runs += 1);
        runs
    }

    fn sized(&self, src: usize, dst: usize, element: usize) -> Result<(), LayoutError> {
        let want_src = self.source_elements as usize * element;
        let want_dst = self.dest_elements as usize * element;
        if src != want_src {
            return Err(LayoutError::Buffer {
                got: src,
                want: want_src,
            });
        }
        if dst != want_dst {
            return Err(LayoutError::Buffer {
                got: dst,
                want: want_dst,
            });
        }
        Ok(())
    }

    /// CPU bytes to CPU bytes: read the source in its logical arrangement and
    /// write the destination in this layout's. Padding is written as zero.
    ///
    /// This is the materialization backend. The GPU fill runs the same walk
    /// into pinned staging; there is no second implementation of the map.
    pub fn apply(&self, source: &[u8], destination: &mut [u8], element: usize) -> Result<(), LayoutError> {
        self.sized(source.len(), destination.len(), element)?;
        self.for_each_run(|to, from, length| {
            let at = to as usize * element;
            let span = length as usize * element;
            match from {
                Some(index) => {
                    let start = index as usize * element;
                    destination[at..at + span].copy_from_slice(&source[start..start + span]);
                }
                None => destination[at..at + span].fill(0),
            }
        });
        Ok(())
    }

    /// The same walk, run backwards: read the destination in this layout's
    /// arrangement and write the source in logical order. Padding is skipped —
    /// it holds no element, which is why a padded record round-trips on its
    /// IMAGE rather than on the whole destination.
    pub fn apply_inverse(
        &self,
        destination: &[u8],
        source: &mut [u8],
        element: usize,
    ) -> Result<(), LayoutError> {
        self.sized(source.len(), destination.len(), element)?;
        self.for_each_run(|to, from, length| {
            if let Some(index) = from {
                let at = to as usize * element;
                let span = length as usize * element;
                let start = index as usize * element;
                source[start..start + span].copy_from_slice(&destination[at..at + span]);
            }
        });
        Ok(())
    }
}

// --- the auto-ratification verifier -----------------------------------------

/// What ratifying one record proved, per probe shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub shape: Vec<u64>,
    pub dest_elements: u64,
    pub padded: bool,
}

/// The verdict on one record.
#[derive(Debug, Clone)]
pub struct Ratification {
    pub handle: String,
    pub probes: Vec<Probe>,
}

const POISON: u8 = 0xA5;
const ELEMENT: usize = 4;

/// AUTO-RATIFY one record: apply the map, apply it inverted, compare bytes.
///
/// Every failure this can catch is a failure that is otherwise SILENT — the
/// tensor keeps its name, its dtype and its shape, and its numbers move. Three
/// things are proved per probe shape, and the third is the one a reviewer
/// cannot do by reading:
///
///  1. every destination element is either a source element or declared
///     padding, and the padding is zero;
///  2. every source element is written EXACTLY once, so nothing is dropped and
///     nothing is written twice;
///  3. the bytes that come back through the inverse are the bytes that went in.
///
/// The source is filled with a distinct pattern per element, so a map that
/// swaps two elements fails (3) rather than passing on symmetry, and the
/// destination is prefilled with a poison byte, so a destination slot the walk
/// never visits fails (1) rather than passing on a lucky zero.
pub fn ratify(record: &LayoutMorphism) -> Result<Ratification, LayoutError> {
    record.check()?;
    let mut probes = Vec::new();
    for shape in probe_shapes(record) {
        let plan = match record.plan(&shape) {
            Ok(plan) => plan,
            // A probe the record refuses is not a failure: an exact
            // factorization legitimately refuses a shape it cannot address,
            // and refusing is the behaviour under test elsewhere.
            Err(_) => continue,
        };
        let source_bytes = plan.source_elements as usize * ELEMENT;
        let mut source = vec![0u8; source_bytes];
        for index in 0..plan.source_elements as usize {
            let tag = (index as u32).wrapping_mul(2_654_435_761).to_le_bytes();
            source[index * ELEMENT..(index + 1) * ELEMENT].copy_from_slice(&tag);
        }
        let mut destination = vec![POISON; plan.dest_elements as usize * ELEMENT];
        plan.apply(&source, &mut destination, ELEMENT)?;

        let mut mapped = 0u64;
        let mut pad_slots = Vec::new();
        plan.for_each(|to, from| match from {
            Some(_) => mapped += 1,
            None => pad_slots.push(to),
        });
        if mapped != plan.source_elements {
            return Err(LayoutError::Record {
                handle: record.handle(),
                detail: format!(
                    "shape {shape:?}: the map writes {mapped} of {} source elements. \
                     A layout that does not write every element exactly once drops \
                     or duplicates numbers with every name and shape still correct",
                    plan.source_elements
                ),
            });
        }
        for slot in &pad_slots {
            let at = *slot as usize * ELEMENT;
            if destination[at..at + ELEMENT] != [0u8; ELEMENT] {
                return Err(LayoutError::Record {
                    handle: record.handle(),
                    detail: format!("shape {shape:?}: padding at {slot} is not zero"),
                });
            }
        }
        if !plan.padded() && !pad_slots.is_empty() {
            return Err(LayoutError::Record {
                handle: record.handle(),
                detail: format!(
                    "shape {shape:?}: {} padding slots in a plan that reports none",
                    pad_slots.len()
                ),
            });
        }

        let mut back = vec![POISON; source_bytes];
        plan.apply_inverse(&destination, &mut back, ELEMENT)?;
        if back != source {
            let at = back
                .iter()
                .zip(&source)
                .position(|(left, right)| left != right)
                .unwrap_or(0);
            return Err(LayoutError::Record {
                handle: record.handle(),
                detail: format!(
                    "shape {shape:?}: the round trip does not return the source \
                     bytes; element {} differs",
                    at / ELEMENT
                ),
            });
        }
        probes.push(Probe {
            shape,
            dest_elements: plan.dest_elements(),
            padded: plan.padded(),
        });
    }
    if probes.is_empty() {
        return Err(LayoutError::Record {
            handle: record.handle(),
            detail: "no probe shape reached this record, so nothing was proved".into(),
        });
    }
    Ok(Ratification {
        handle: record.handle(),
        probes,
    })
}

/// Probe shapes DERIVED FROM THE RECORD, never hand-fed.
///
/// A hand-written probe list is a list somebody keeps in step with the catalog,
/// and the day they forget is the day a new record is ratified against nothing.
/// The alignment each axis needs is the product of its LITERAL factors — an
/// exact division by 128 refuses anything that is not a multiple of 128 — so
/// the aligned probe is that product times a small distinct multiplier, and the
/// ragged probe is one element short of it. A record whose factorization is
/// exact refuses the ragged probe, which is correct and is why `ratify` skips
/// rather than fails on a refused shape.
pub fn probe_shapes(record: &LayoutMorphism) -> Vec<Vec<u64>> {
    if record.is_identity() {
        return (1..=4).map(|rank| (0..rank).map(|at| at + 2).collect()).collect();
    }
    let mut alignment = vec![1u64; record.rank];
    for sub in &record.sub_axes {
        if let Ok(literal) = sub.extent.trim().parse::<u64>() {
            alignment[sub.axis] *= literal;
        }
    }
    let aligned: Vec<u64> = alignment
        .iter()
        .enumerate()
        .map(|(at, &align)| align * (at as u64 + 2))
        .collect();
    let ragged: Vec<u64> = aligned
        .iter()
        .map(|&dim| if dim > 1 { dim - 1 } else { dim })
        .collect();
    if ragged == aligned {
        vec![aligned]
    } else {
        vec![aligned, ragged]
    }
}

// --- the vendored catalog ---------------------------------------------------

fn parsed_catalog() -> &'static Result<BTreeMap<String, LayoutMorphism>, LayoutError> {
    static PARSED: OnceLock<Result<BTreeMap<String, LayoutMorphism>, LayoutError>> =
        OnceLock::new();
    PARSED.get_or_init(|| {
        let mut out = BTreeMap::new();
        for (file, document) in CATALOG {
            let record: LayoutMorphism =
                serde_json::from_str(document).map_err(|error| LayoutError::Catalog {
                    file: (*file).into(),
                    detail: error.to_string(),
                })?;
            record.check()?;
            if out.insert(record.handle(), record).is_some() {
                return Err(LayoutError::Catalog {
                    file: (*file).into(),
                    detail: "declares a handle the catalog already carries".into(),
                });
            }
        }
        if !out.contains_key(DEFAULT_LAYOUT) {
            return Err(LayoutError::Catalog {
                file: "spec/v2/layouts".into(),
                detail: format!("carries no {DEFAULT_LAYOUT} record"),
            });
        }
        Ok(out)
    })
}

/// The vendored catalog, parsed and self-checked once.
pub fn catalog() -> Result<&'static BTreeMap<String, LayoutMorphism>, LayoutError> {
    match parsed_catalog() {
        Ok(records) => Ok(records),
        Err(error) => Err(error.clone()),
    }
}

/// Look one record up by `<name>@<version>`.
pub fn arrangement(handle: &str) -> Result<&'static LayoutMorphism, LayoutError> {
    catalog()?.get(handle).ok_or_else(|| LayoutError::Record {
        handle: handle.into(),
        detail: "the catalog carries no such arrangement".into(),
    })
}

// --- the shape formula ------------------------------------------------------
//
// The same tiny language `v2doc.go` evaluates: integers, `dN`, `*`, `/` and
// `ceil(...)`. Division is EXACT outside `ceil()`; a record that divides an
// axis it cannot divide is refusing a shape, not rounding it.
//
// This is a second EVALUATOR of one language, which the tfm1 format already
// establishes as the price of two languages reading one wire format. It is
// held to the Go one by banked vectors (`spec/v2/vectors/layout-plans.json`),
// checked by tests on both sides: a drift between them fails there.

fn eval(text: &str, shape: &[u64]) -> Result<u64, String> {
    eval_chain(&text.replace(' ', ""), shape, false)
}

fn eval_chain(text: &str, shape: &[u64], ceiling: bool) -> Result<u64, String> {
    let mut value: u64 = 1;
    let mut divide = false;
    let mut rest = text;
    if rest.is_empty() {
        return Err("empty formula".into());
    }
    loop {
        let operand;
        if let Some(inner) = rest.strip_prefix("ceil(") {
            let mut depth = 1usize;
            let mut end = 0usize;
            for (at, character) in inner.char_indices() {
                match character {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = at;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth != 0 {
                return Err(format!("{text:?} has an unclosed ceil("));
            }
            operand = eval_chain(&inner[..end], shape, true)?;
            rest = &inner[end + 1..];
        } else {
            let end = rest.find(['*', '/']).unwrap_or(rest.len());
            let token = &rest[..end];
            operand = if let Some(digits) = token.strip_prefix('d') {
                let axis: usize = digits
                    .parse()
                    .map_err(|_| format!("{token:?} is not an axis"))?;
                *shape
                    .get(axis)
                    .ok_or_else(|| format!("{text:?} reads axis {axis} of a rank-{} tensor", shape.len()))?
            } else {
                let literal: u64 = token
                    .parse()
                    .map_err(|_| format!("{token:?} is not a positive integer"))?;
                if literal == 0 {
                    return Err(format!("{token:?} is not a positive integer"));
                }
                literal
            };
            rest = &rest[end..];
        }
        if divide {
            if operand == 0 {
                return Err(format!("{text:?} divides by zero"));
            }
            if ceiling {
                value = value.div_ceil(operand);
            } else {
                if value % operand != 0 {
                    return Err(format!(
                        "{text:?} divides {value} by {operand}, which is not exact"
                    ));
                }
                value /= operand;
            }
        } else {
            value *= operand;
        }
        match rest.chars().next() {
            None => return Ok(value),
            Some('*') => divide = false,
            Some('/') => divide = true,
            Some(other) => return Err(format!("{text:?} has a stray {other:?}")),
        }
        rest = &rest[1..];
        if rest.is_empty() {
            return Err(format!("{text:?} ends on an operator"));
        }
    }
}
