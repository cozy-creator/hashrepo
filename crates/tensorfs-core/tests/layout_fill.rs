//! THE FILL PATH on the CPU: chunks in, arranged bytes out, and the measured
//! facts that decide how the GPU leg has to work.
//!
//! Everything here runs on a box with no card, which is the point of the seam:
//! the transform is the part tensorfs owns and it is provable without a device.

use tensorfs_core::layout_fill::{ChunkedSource, FillSink, HostSink, TensorFill, fill};
use tensorfs_core::layout_morphism::{LayoutMorphism, arrangement};

fn record(handle: &str) -> &'static LayoutMorphism {
    arrangement(handle).unwrap_or_else(|error| panic!("{handle}: {error}"))
}

/// A tensor's bytes, one distinct 4-byte pattern per element.
fn tensor(elements: usize) -> Vec<u8> {
    (0..elements as u32)
        .flat_map(|at| at.wrapping_mul(2_654_435_761).to_le_bytes())
        .collect()
}

/// Slice a buffer into chunks of the given sizes. Sizes that are NOT multiples
/// of the element size are the interesting ones: chunk lengths are a storage
/// decision and know nothing about elements.
fn chunked<'a>(bytes: &'a [u8], sizes: &[usize]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let mut at = 0usize;
    for &size in sizes {
        let end = (at + size).min(bytes.len());
        out.push(&bytes[at..end]);
        at = end;
    }
    if at < bytes.len() {
        out.push(&bytes[at..]);
    }
    out
}

/// The reference: the same arrangement applied to one contiguous buffer by the
/// applier `layout_morphism` already ships. If the fill path and the reference
/// ever disagree, the chunked read is inventing bytes.
fn reference(handle: &str, shape: &[u64], source: &[u8]) -> Vec<u8> {
    let plan = record(handle).plan(shape).unwrap();
    let mut out = vec![0u8; plan.dest_elements() as usize * 4];
    plan.apply(source, &mut out, 4).unwrap();
    out
}

#[test]
fn the_fill_path_and_the_reference_applier_agree() {
    let shape = [2u64, 3, 4, 5];
    let source = tensor(2 * 3 * 4 * 5);
    let chunks = chunked(&source, &[64, 64, 64]);
    let reader = ChunkedSource::new(&chunks);
    let plan = record("torch.channels_last-2d@1").plan(&shape).unwrap();
    let mut sink = HostSink::with_capacity(plan.dest_elements() as usize * 4);
    let stats = fill(
        &reader,
        &TensorFill {
            layout: record("torch.channels_last-2d@1"),
            shape: shape.to_vec(),
            element_bytes: 4,
            device_offset: 0,
        },
        &mut sink,
    )
    .expect("fill");

    assert_eq!(
        sink.bytes(),
        reference("torch.channels_last-2d@1", &shape, &source).as_slice(),
        "the fill path and the reference applier disagree"
    );
    assert_eq!(stats.source_bytes, source.len() as u64);
    assert_eq!(stats.destination_bytes, source.len() as u64);
    assert_eq!(stats.padding_bytes, 0);
}

/// THE PROPERTY THAT MATTERS FOR A REAL TREE: chunk boundaries are a storage
/// decision. A tensor split at 7-byte boundaries — straddling elements, ragged
/// against every axis — must produce the same bytes as one whole buffer.
#[test]
fn chunk_boundaries_do_not_change_the_result() {
    let shape = [4u64, 6];
    let source = tensor(24);
    let whole = reference("torch.transposed@1", &shape, &source);

    for sizes in [
        vec![96],
        vec![48, 48],
        vec![7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5],
        vec![1, 95],
        vec![95, 1],
    ] {
        let chunks = chunked(&source, &sizes);
        let reader = ChunkedSource::new(&chunks);
        let mut sink = HostSink::with_capacity(whole.len());
        let stats = fill(
            &reader,
            &TensorFill {
                layout: record("torch.transposed@1"),
                shape: shape.to_vec(),
                element_bytes: 4,
                device_offset: 0,
            },
            &mut sink,
        )
        .unwrap_or_else(|error| panic!("chunking {sizes:?}: {error}"));
        assert_eq!(
            sink.bytes(),
            whole.as_slice(),
            "chunking {sizes:?} changed the arranged bytes"
        );
        assert!(stats.chunks >= sizes.len() as u64 - 1 || sizes.len() == 1);
    }
}

/// THE MEASURED FACT THE GPU LEG IS DESIGNED AROUND.
///
/// A morphism's contiguous-run count decides whether it could ride a
/// scatter-gather copy engine or has to be applied while the bytes are already
/// being touched. These numbers are computed from the arrangement, and they are
/// asserted exactly, because the whole shape of the fill path follows from
/// them: one of these arrangements is free and one of them is forty million
/// transfers if you get it wrong.
#[test]
fn the_run_count_says_which_arrangements_can_ride_a_copy() {
    // The identity is ONE run at any shape: a whole tensor is a single DMA and
    // the fill path costs nothing at all.
    let identity = record("torch.contiguous@1").plan(&[64u64, 128]).unwrap();
    assert_eq!(identity.run_count(), 1, "the identity is not one run");

    // channels_last is one run PER ELEMENT: its innermost storage axis is the
    // channel, whose source stride is H*W, so consecutive destination elements
    // are 20 elements apart in the source. Nothing folds.
    //
    // THIS COUNT IS STRUCTURAL, and the distinction cost a rewrite to learn. An
    // earlier walk discovered runs by visiting elements and noticing adjacency,
    // and measured 119 here — because at the image boundary channels_last's
    // last element really does sit next to the next image's first. That number
    // was correct and useless: it drifts with the shape's accidental
    // adjacencies, and computing it costs one odometer step per element, which
    // measured 0.02 GiB/s on a card for what is a single memcpy plus a DMA. The
    // decomposition is now derived from the strides, and the count is a
    // property of the ARRANGEMENT.
    let nhwc = record("torch.channels_last-2d@1")
        .plan(&[2u64, 3, 4, 5])
        .unwrap();
    assert_eq!(
        nhwc.run_count(),
        nhwc.dest_elements(),
        "channels_last folded sub-axes it cannot fold"
    );

    // The blocked scale layout sits between: its innermost storage sub-axis IS
    // innermost in the source, so runs are exactly four elements long.
    let blocked = record("cublas.blockscale-128x4@1")
        .plan(&[256u64, 12])
        .unwrap();
    assert_eq!(blocked.run_count(), blocked.dest_elements() / 4);

    // And a padded plan folds NOTHING, because a run spanning padding would be
    // part copy and part zero.
    let ragged = record("cublas.blockscale-128x4@1")
        .plan(&[100u64, 3])
        .unwrap();
    assert_eq!(ragged.run_count(), ragged.dest_elements());

    // And the stats report it, so an operator reading a fill does not have to
    // re-derive any of this.
    let source = tensor(2 * 3 * 4 * 5);
    let chunks = chunked(&source, &[240, 240]);
    let reader = ChunkedSource::new(&chunks);
    let mut sink = HostSink::with_capacity(nhwc.dest_elements() as usize * 4);
    let stats = fill(
        &reader,
        &TensorFill {
            layout: record("torch.channels_last-2d@1"),
            shape: vec![2, 3, 4, 5],
            element_bytes: 4,
            device_offset: 0,
        },
        &mut sink,
    )
    .unwrap();
    assert!(
        stats.runs_per_element(4) == 1.0,
        "the stats do not report a per-element gather as one"
    );
}

/// Padding is budget the caller paid for, so it is measured and reported rather
/// than quietly written.
#[test]
fn a_padded_arrangement_reports_the_bytes_it_invented() {
    let shape = [100u64, 3];
    let source = tensor(300);
    let chunks = chunked(&source, &[600, 600]);
    let reader = ChunkedSource::new(&chunks);
    let plan = record("cublas.blockscale-128x4@1").plan(&shape).unwrap();
    let mut sink = HostSink::with_capacity(plan.dest_elements() as usize * 4);
    let stats = fill(
        &reader,
        &TensorFill {
            layout: record("cublas.blockscale-128x4@1"),
            shape: shape.to_vec(),
            element_bytes: 4,
            device_offset: 0,
        },
        &mut sink,
    )
    .unwrap();
    assert_eq!(stats.source_bytes, 300 * 4);
    assert_eq!(stats.destination_bytes, 128 * 4 * 4);
    assert_eq!(
        stats.padding_bytes,
        stats.destination_bytes - stats.source_bytes,
        "every byte that is not a source byte is padding, and it is all counted"
    );
    assert_eq!(
        sink.bytes(),
        reference("cublas.blockscale-128x4@1", &shape, &source).as_slice()
    );
}

/// Several tensors into ONE destination at the offsets the caller chose. This
/// is the shape of a real load: varena decides where each tensor lands and
/// tensorfs never picks an address.
#[test]
fn tensors_land_at_the_offsets_the_caller_chose() {
    let shape = [2u64, 4];
    let first = tensor(8);
    let second: Vec<u8> = tensor(8).iter().map(|b| b ^ 0xFF).collect();
    let want_first = reference("torch.transposed@1", &shape, &first);
    let want_second = reference("torch.transposed@1", &shape, &second);

    let mut sink = HostSink::with_capacity(want_first.len() + want_second.len());
    for (bytes, offset) in [(&first, 0u64), (&second, want_first.len() as u64)] {
        let chunks = chunked(bytes, &[32]);
        let reader = ChunkedSource::new(&chunks);
        fill(
            &reader,
            &TensorFill {
                layout: record("torch.transposed@1"),
                shape: shape.to_vec(),
                element_bytes: 4,
                device_offset: offset,
            },
            &mut sink,
        )
        .unwrap();
    }
    assert_eq!(&sink.bytes()[..want_first.len()], want_first.as_slice());
    assert_eq!(&sink.bytes()[want_first.len()..], want_second.as_slice());
}

/// RED ARMS. Each one is a way for a fill to be silently wrong, and each one
/// refuses instead.
#[test]
fn a_fill_refuses_rather_than_inventing_bytes() {
    let shape = vec![2u64, 4];
    let source = tensor(8);

    // A source that is not the size the shape says. Under-length is the bad
    // one: it would read whatever follows in the chunk list.
    let short = tensor(7);
    let chunks = chunked(&short, &[28]);
    let reader = ChunkedSource::new(&chunks);
    let mut sink = HostSink::with_capacity(32);
    assert!(
        fill(
            &reader,
            &TensorFill {
                layout: record("torch.transposed@1"),
                shape: shape.clone(),
                element_bytes: 4,
                device_offset: 0,
            },
            &mut sink,
        )
        .is_err(),
        "a short source filled anyway"
    );

    // A destination that does not have room at the offset asked for.
    let chunks = chunked(&source, &[32]);
    let reader = ChunkedSource::new(&chunks);
    let mut small = HostSink::with_capacity(32);
    assert!(
        fill(
            &reader,
            &TensorFill {
                layout: record("torch.transposed@1"),
                shape: shape.clone(),
                element_bytes: 4,
                device_offset: 16,
            },
            &mut small,
        )
        .is_err(),
        "a fill ran off the end of the destination"
    );

    // A rank the arrangement is not defined for.
    let mut sink = HostSink::with_capacity(64);
    assert!(
        fill(
            &reader,
            &TensorFill {
                layout: record("torch.channels_last-2d@1"),
                shape: vec![2, 4],
                element_bytes: 4,
                device_offset: 0,
            },
            &mut sink,
        )
        .is_err(),
        "a rank-4 arrangement filled a rank-2 tensor"
    );
}

/// An unratified candidate never moves a byte, in the crate a worker actually
/// calls and not only in the decision engine.
#[test]
fn an_unratified_candidate_never_fills() {
    let mut wish: LayoutMorphism = record("torch.transposed@1").clone();
    wish.candidate = true;
    let source = tensor(8);
    let chunks = chunked(&source, &[32]);
    let reader = ChunkedSource::new(&chunks);
    let mut sink = HostSink::with_capacity(32);
    let error = fill(
        &reader,
        &TensorFill {
            layout: &wish,
            shape: vec![2, 4],
            element_bytes: 4,
            device_offset: 0,
        },
        &mut sink,
    )
    .expect_err("an unratified candidate filled a destination");
    assert!(error.to_string().contains("candidate"), "{error}");
}

/// A sink that refuses to stage must stop the fill, not be filled around. The
/// GPU sink refuses a tensor larger than its pinned staging budget, and this
/// proves the caller of `fill` honours that refusal without a card in the box.
#[test]
fn a_sink_that_refuses_staging_stops_the_fill() {
    struct Refusing;
    impl FillSink for Refusing {
        fn staging(
            &mut self,
            bytes: usize,
            _offset: u64,
        ) -> Result<&mut [u8], tensorfs_core::layout_morphism::LayoutError> {
            Err(tensorfs_core::layout_morphism::LayoutError::Buffer {
                got: 0,
                want: bytes,
            })
        }
        fn commit(&mut self) -> Result<(), tensorfs_core::layout_morphism::LayoutError> {
            panic!("commit ran after staging refused");
        }
    }
    let source = tensor(8);
    let chunks = chunked(&source, &[32]);
    let reader = ChunkedSource::new(&chunks);
    assert!(
        fill(
            &reader,
            &TensorFill {
                layout: record("torch.transposed@1"),
                shape: vec![2, 4],
                element_bytes: 4,
                device_offset: 0,
            },
            &mut Refusing,
        )
        .is_err()
    );
}
