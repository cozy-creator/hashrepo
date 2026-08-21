//! THE FILL PATH: chunks on disk to bytes at a destination address, with the
//! morphism applied on the way.
//!
//! `layout_morphism` decides which byte goes where. This module moves them, and
//! it moves them into memory SOMEBODY ELSE OWNS. That is the seam and it is not
//! negotiable: varena owns the address space — reservations, backing, stable
//! VAs, paging — and hands out destination pointers; tensorfs fills them.
//! Nothing here allocates device memory, and nothing here decides what to load.
//!
//! ## Why the destination is a trait and not a CUDA call
//!
//! varena's refill plane already owns the byte-motion mechanics: io_uring on the
//! disk leg, budgeted pinned host slabs, async H2D on a copy stream, chunk
//! recycling. Writing a second one here would be the duplication the
//! one-implementation rule exists to prevent, and it would be a WORSE one,
//! because the measured 93-99% H2D-bound refill is that engine.
//!
//! So the split is: tensorfs owns the PLAN and the TRANSFORM — the part varena
//! structurally cannot do, because a morphism is not a contiguous range map —
//! and the destination is a [`FillSink`] the caller provides. varena implements
//! it over a slab plus its own H2D; [`HostSink`] implements it over plain
//! memory and is the CPU materialization backend; the `cuda` feature ships a
//! reference [`CudaSink`] so this crate can prove the device leg on its own
//! card tests rather than asserting it.
//!
//! ## Why the transform runs during the host staging pass
//!
//! A morphism decomposes into contiguous runs, and the run count is COMPUTED
//! ([`Plan::for_each_run`]), never assumed. The identity is one run — a whole
//! tensor is a single DMA and the fill path costs nothing. `channels_last` is
//! one run PER ELEMENT, because its innermost storage axis has source stride
//! H*W: handing that to a scatter-gather copy engine is forty million two-byte
//! transfers. The bytes are already being copied once, disk to pinned staging,
//! so the permutation rides THAT copy and the device leg stays one contiguous
//! DMA. `runs_per_element` in the stats is what tells an operator which case a
//! given tensor was.

use crate::layout_morphism::{LayoutError, LayoutMorphism, Plan};

/// Where filled bytes go. Implemented by varena over a reservation, by
/// [`HostSink`] over plain memory, and by `CudaSink` over a device pointer.
///
/// The two calls are deliberately ordered: the caller of `fill` asks for
/// staging FIRST, writes the arranged bytes into it, and commits. A sink whose
/// staging IS the destination (the host backend) commits for free; one whose
/// destination is a device pointer commits with one contiguous transfer.
pub trait FillSink {
    /// Hand back at least `bytes` of writable staging for the tensor destined
    /// for `device_offset`. The buffer's contents are undefined on entry and
    /// the caller writes all `bytes` of it.
    fn staging(&mut self, bytes: usize, device_offset: u64) -> Result<&mut [u8], LayoutError>;

    /// Move the staged bytes to their destination. After this returns the
    /// staging buffer may be recycled.
    fn commit(&mut self) -> Result<(), LayoutError>;
}

/// What one tensor's fill is: its source bytes' logical shape, its element
/// size, the arrangement the destination wants, and where it lands.
#[derive(Debug, Clone)]
pub struct TensorFill<'a> {
    /// The arrangement the DESTINATION wants. `torch.contiguous@1` means the
    /// bytes are copied unchanged, which is the fast path and still goes
    /// through here so there is one path and not two.
    pub layout: &'a LayoutMorphism,
    pub shape: Vec<u64>,
    pub element_bytes: usize,
    /// Byte offset into the caller's destination. varena computes it; tensorfs
    /// never chooses an address.
    pub device_offset: u64,
}

/// What a fill actually did. Every field is measured during the fill, because
/// "the morphism was free" is a claim and not a design property.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FillStats {
    /// Source bytes read out of the chunks.
    pub source_bytes: u64,
    /// Bytes written to the destination — larger than `source_bytes` exactly
    /// when the arrangement pads.
    pub destination_bytes: u64,
    /// Zero-filled padding bytes. Meaningless data the kernel never reads, and
    /// budget the caller paid for, so it is reported rather than hidden.
    pub padding_bytes: u64,
    /// Contiguous runs the morphism decomposed into. 1 means the tensor was one
    /// DMA; `destination_elements` means every element moved on its own.
    pub runs: u64,
    /// Source chunks touched.
    pub chunks: u64,
}

impl FillStats {
    /// Runs per destination element: 0.0 is unreachable, ~0.0 is a whole-tensor
    /// copy, 1.0 is a per-element gather. The one number that says whether this
    /// arrangement could ever have ridden a scatter-gather engine.
    pub fn runs_per_element(&self, element_bytes: usize) -> f64 {
        let elements = self.destination_bytes / element_bytes.max(1) as u64;
        if elements == 0 {
            return 0.0;
        }
        self.runs as f64 / elements as f64
    }
}

/// One tensor's source bytes, as the manifest lists them: an ordered chunk
/// list that is logically one contiguous buffer.
///
/// The chunks are NOT concatenated first. A fill that materialized the source
/// before rearranging it would double the host bytes touched, which on a 635
/// MiB conv weight set is the whole cost the arrangement exists to remove.
pub struct ChunkedSource<'a> {
    chunks: &'a [&'a [u8]],
    /// Start offset of each chunk in the logical buffer, plus the total.
    starts: Vec<u64>,
}

impl<'a> ChunkedSource<'a> {
    pub fn new(chunks: &'a [&'a [u8]]) -> Self {
        let mut starts = Vec::with_capacity(chunks.len() + 1);
        let mut at = 0u64;
        for chunk in chunks {
            starts.push(at);
            at += chunk.len() as u64;
        }
        starts.push(at);
        Self { chunks, starts }
    }

    pub fn len(&self) -> u64 {
        *self.starts.last().unwrap_or(&0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole source as one slice, when it already is one.
    ///
    /// The common case: a tensor usually lives inside one chunk, and the
    /// chunked reader pays a binary search and a bounds check PER RUN — which
    /// on a per-element gather is per element. MEASURED on the 4070, release,
    /// a [256, 128, 3, 3] channels_last fill: 15.5 ms through the chunked
    /// reader against 11.3 ms through this path, both single samples taken
    /// before the repeat harness existed. Worth having, and NOT the dominant
    /// cost of a per-element gather — that is the gather itself, which the same
    /// leg now measures at 9.1 ms best-of-10, about 31 ns per element.
    pub fn contiguous(&self) -> Option<&'a [u8]> {
        match self.chunks {
            [only] => Some(only),
            _ => None,
        }
    }

    /// Copy `out.len()` bytes starting at logical `offset`, crossing chunk
    /// boundaries as needed. A run that straddles two chunks is normal: chunk
    /// lengths are a storage decision and know nothing about element size.
    fn read_at(&self, offset: u64, out: &mut [u8], touched: &mut u64) -> Result<(), LayoutError> {
        if offset + out.len() as u64 > self.len() {
            return Err(LayoutError::Buffer {
                got: self.len() as usize,
                want: (offset + out.len() as u64) as usize,
            });
        }
        let mut index = match self.starts.binary_search(&offset) {
            Ok(at) => at,
            Err(at) => at - 1,
        };
        let mut written = 0usize;
        let mut cursor = offset;
        while written < out.len() {
            let chunk = self.chunks[index];
            let within = (cursor - self.starts[index]) as usize;
            let take = (chunk.len() - within).min(out.len() - written);
            out[written..written + take].copy_from_slice(&chunk[within..within + take]);
            written += take;
            cursor += take as u64;
            *touched += 1;
            index += 1;
        }
        Ok(())
    }
}

/// FILL ONE TENSOR: read its chunks, apply the arrangement, hand the result to
/// the sink.
///
/// This is the ONE fill. The CPU materialization backend and the GPU fill are
/// the same call with a different sink — there is no second walk, no second
/// index map, and no place for the two to disagree about which byte goes where.
pub fn fill<S: FillSink>(
    source: &ChunkedSource<'_>,
    tensor: &TensorFill<'_>,
    sink: &mut S,
) -> Result<FillStats, LayoutError> {
    tensor.layout.ensure_ratified()?;
    if tensor.element_bytes == 0 {
        return Err(LayoutError::Buffer { got: 0, want: 1 });
    }
    let plan: Plan = tensor.layout.plan(&tensor.shape)?;
    let want_source = plan.source_elements() * tensor.element_bytes as u64;
    if source.len() != want_source {
        return Err(LayoutError::Buffer {
            got: source.len() as usize,
            want: want_source as usize,
        });
    }
    let destination_bytes = plan.dest_elements() * tensor.element_bytes as u64;
    let staging = sink.staging(destination_bytes as usize, tensor.device_offset)?;

    let element = tensor.element_bytes;
    let mut stats = FillStats {
        source_bytes: want_source,
        destination_bytes,
        ..FillStats::default()
    };

    // ONE chunk: hand the walk a slice it can index and let it run. Same walk,
    // same runs, same bytes — the difference is a binary search per run that
    // this case does not have to pay.
    if let Some(whole) = source.contiguous() {
        plan.apply(whole, staging, element)?;
        plan.for_each_run(|_, from, length| {
            stats.runs += 1;
            if from.is_none() {
                stats.padding_bytes += length * element as u64;
            }
        });
        stats.chunks = 1;
        sink.commit()?;
        return Ok(stats);
    }
    // The runs are visited in DESTINATION order, so the staging buffer is
    // written strictly forward — the access pattern a pinned buffer and a
    // following DMA both want.
    let mut failure: Option<LayoutError> = None;
    plan.for_each_run(|to, from, length| {
        if failure.is_some() {
            return;
        }
        stats.runs += 1;
        let at = to as usize * element;
        let span = length as usize * element;
        match from {
            Some(index) => {
                if let Err(error) = source.read_at(
                    index * element as u64,
                    &mut staging[at..at + span],
                    &mut stats.chunks,
                ) {
                    failure = Some(error);
                }
            }
            None => {
                staging[at..at + span].fill(0);
                stats.padding_bytes += span as u64;
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    sink.commit()?;
    Ok(stats)
}

/// The CPU backend: the destination is plain host memory the caller owns.
///
/// This is what materializes a derived tree the storage tier decided to write
/// down, and it is the reference the GPU fill is byte-compared against.
pub struct HostSink {
    buffer: Vec<u8>,
    pending: Option<(usize, usize)>,
}

impl HostSink {
    /// A destination of exactly `bytes`, zeroed.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buffer: vec![0u8; bytes],
            pending: None,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buffer
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }
}

impl FillSink for HostSink {
    fn staging(&mut self, bytes: usize, device_offset: u64) -> Result<&mut [u8], LayoutError> {
        let at = device_offset as usize;
        if at + bytes > self.buffer.len() {
            return Err(LayoutError::Buffer {
                got: self.buffer.len(),
                want: at + bytes,
            });
        }
        self.pending = Some((at, bytes));
        Ok(&mut self.buffer[at..at + bytes])
    }

    /// Nothing to move: the staging buffer IS the destination. The host backend
    /// is the one case where the transform is the whole cost.
    fn commit(&mut self) -> Result<(), LayoutError> {
        self.pending = None;
        Ok(())
    }
}
