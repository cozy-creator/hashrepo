//! THE DEVICE LEG, on a real card. A few MiB at a time, on device 0.
//!
//! Skipped unless the run is SANCTIONED. `cargo test` must stay CPU-safe by
//! default, because the development box's card is shared and arbitrated by a
//! human coordinator. The gate is `TENSORFS_GPU_WINDOW=1`, exported for the
//! life of a granted window and unset at handback; `TENSORFS_GPU=1` is accepted
//! for exclusive pods where nothing is arbitrated. Same two-spellings-one-rule
//! shape varena's `tests/gpu.rs` settled on, and for the same reason: a gate
//! with two DIFFERENT names is a gate with a hole.
//!
//! A SKIP SAYS SO. Every skip prints why, so a green run with no card is
//! visibly a green run with no card and not a proof that anything worked.
//!
//! THE TEST ALLOCATES THE DEVICE MEMORY, NOT THE CRATE. That is the seam under
//! test: varena owns the address space and hands over a pointer. Here the test
//! plays varena — deliberately, in the test, using the driver symbols the crate
//! exposes for exactly this — and the crate only ever writes to what it is
//! given.

use std::ffi::c_void;

use tensorfs_core::layout_fill::{ChunkedSource, HostSink, TensorFill, fill};
use tensorfs_core::layout_morphism::arrangement;
use tensorfs_cuda::{CudaSink, CUdeviceptr, bind, driver};

const SANCTIONING_VARS: [&str; 2] = ["TENSORFS_GPU_WINDOW", "TENSORFS_GPU"];

/// Is this run sanctioned? Pure and takes its own lookup, so the proof of the
/// gate never needs a card and never races another test over process-global
/// environment state.
fn sanctioned(lookup: impl Fn(&str) -> Option<String>) -> bool {
    SANCTIONING_VARS
        .iter()
        .any(|name| lookup(name).as_deref() == Some("1"))
}

fn gpu() -> bool {
    sanctioned(|name| std::env::var(name).ok())
}

fn tensor(elements: usize) -> Vec<u8> {
    (0..elements as u32)
        .flat_map(|at| at.wrapping_mul(2_654_435_761).to_le_bytes())
        .collect()
}

/// ACCEPTANCE (a): one morphism, defined once as data, executed by BOTH
/// backends, byte-verified equal.
///
/// The host backend arranges into memory; the device backend arranges into
/// pinned staging and DMAs to a device pointer. The bytes read back off the
/// card must equal the host backend's, byte for byte — not "close", not
/// "the same shape". One walk, two destinations.
#[test]
fn both_backends_produce_the_same_bytes_for_channels_last() {
    if !gpu() {
        eprintln!(
            "skipped: shared GPU — set TENSORFS_GPU_WINDOW=1 for a granted window. \
             NOTHING ON A CARD WAS PROVED BY THIS RUN."
        );
        return;
    }
    bind(0).expect("bind device 0");
    let cuda = driver().expect("driver");

    // A conv-weight-shaped tensor: 256 filters, 128 channels, 3x3.
    let shape = vec![256u64, 128, 3, 3];
    let elements: u64 = shape.iter().product();
    let source = tensor(elements as usize);
    let layout = arrangement("torch.channels_last-2d@1").expect("record");
    let plan = layout.plan(&shape).expect("plan");
    let bytes = plan.dest_elements() as usize * 4;

    // The host backend, as the reference.
    let chunks = [&source[..]];
    let mut host = HostSink::with_capacity(bytes);
    let host_stats = fill(
        &ChunkedSource::new(&chunks),
        &TensorFill {
            layout,
            shape: shape.clone(),
            element_bytes: 4,
            device_offset: 0,
        },
        &mut host,
    )
    .expect("host fill");

    // THE TEST PLAYS VARENA: it owns the destination.
    let mut destination: CUdeviceptr = 0;
    unsafe {
        assert_eq!(
            (cuda.mem_alloc)(&mut destination, bytes),
            0,
            "the test could not allocate its own scratch destination"
        );
    }
    let mut device = CudaSink::new(destination, bytes).expect("pinned staging");
    // Best of 10, for the same reason the identity leg takes ten.
    let mut best = f64::INFINITY;
    let mut device_stats = Default::default();
    for _ in 0..10 {
        let started = std::time::Instant::now();
        device_stats = fill(
            &ChunkedSource::new(&chunks),
            &TensorFill {
                layout,
                shape: shape.clone(),
                element_bytes: 4,
                device_offset: 0,
            },
            &mut device,
        )
        .expect("device fill");
        best = best.min(started.elapsed().as_secs_f64());
    }
    let elapsed = std::time::Duration::from_secs_f64(best);

    let mut readback = vec![0u8; bytes];
    unsafe {
        assert_eq!(
            (cuda.memcpy_dtoh)(readback.as_mut_ptr() as *mut c_void, destination, bytes),
            0,
            "read back"
        );
        let _ = (cuda.mem_free)(destination);
    }

    assert_eq!(
        readback,
        host.bytes(),
        "the two backends of ONE transform implementation produced different bytes"
    );
    assert_eq!(host_stats, device_stats, "the two backends measured differently");
    // Ten reps, ten transfers: the sink counts what it actually moved, which is
    // the number a bench reads and so must not be a per-call reset.
    assert_eq!(device.bytes_to_device, bytes as u64 * 10);
    eprintln!(
        "tensorfs#154 (a): {} elements, {} bytes, {} runs — host and device \
         backends byte-identical. Device fill (per-element gather into pinned \
         staging + one H2D): {:.3} ms = {:.2} GiB/s",
        elements,
        bytes,
        host_stats.runs,
        elapsed.as_secs_f64() * 1e3,
        bytes as f64 / elapsed.as_secs_f64() / (1u64 << 30) as f64
    );
}

/// The identity is the fast path and it must be visibly fast: one run, one
/// contiguous DMA, no per-element anything. The number is printed so the
/// bench leg has a floor to compare against rather than a memory of one.
#[test]
fn the_identity_fill_is_one_run_and_one_transfer() {
    if !gpu() {
        eprintln!(
            "skipped: shared GPU — set TENSORFS_GPU_WINDOW=1 for a granted window. \
             NOTHING ON A CARD WAS PROVED BY THIS RUN."
        );
        return;
    }
    bind(0).expect("bind device 0");
    let cuda = driver().expect("driver");

    let megabytes = 16usize;
    let bytes = megabytes << 20;
    let source = vec![0x5Au8; bytes];
    let layout = arrangement("torch.contiguous@1").expect("record");

    let mut destination: CUdeviceptr = 0;
    unsafe {
        assert_eq!((cuda.mem_alloc)(&mut destination, bytes), 0, "alloc");
    }
    let mut device = CudaSink::new(destination, bytes).expect("pinned staging");
    let chunks = [&source[..]];
    // REPEATED, and the BEST is reported. This box's card drives a desktop, so
    // a single sample carries the compositor's noise: three consecutive single
    // samples of this exact fill spread 2.8/4.4/5.6 ms, which is a 2x spread on
    // a number a bench floor is supposed to be compared against.
    let mut stats = Default::default();
    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let started = std::time::Instant::now();
        stats = fill(
            &ChunkedSource::new(&chunks),
            &TensorFill {
                layout,
                shape: vec![(bytes / 4) as u64],
                element_bytes: 4,
                device_offset: 0,
            },
            &mut device,
        )
        .expect("fill");
        best = best.min(started.elapsed().as_secs_f64());
    }
    let elapsed = std::time::Duration::from_secs_f64(best);

    let mut readback = vec![0u8; 4096];
    unsafe {
        assert_eq!(
            (cuda.memcpy_dtoh)(readback.as_mut_ptr() as *mut c_void, destination, 4096),
            0,
            "read back"
        );
        let _ = (cuda.mem_free)(destination);
    }
    assert!(readback.iter().all(|byte| *byte == 0x5A), "the card holds other bytes");
    assert_eq!(stats.runs, 1, "the identity is not one run");
    eprintln!(
        "tensorfs#154 identity fill: {megabytes} MiB in {:.3} ms = {:.2} GiB/s \
         (staging + H2D, one run, best of 10)",
        elapsed.as_secs_f64() * 1e3,
        bytes as f64 / elapsed.as_secs_f64() / (1u64 << 30) as f64
    );
}

/// A tensor larger than the pinned staging budget is a REFUSAL, and this needs
/// a card only to allocate the buffer — the behaviour under test is the
/// refusal, not the transfer.
#[test]
fn a_tensor_larger_than_the_staging_budget_refuses() {
    if !gpu() {
        eprintln!(
            "skipped: shared GPU — set TENSORFS_GPU_WINDOW=1 for a granted window. \
             NOTHING ON A CARD WAS PROVED BY THIS RUN."
        );
        return;
    }
    bind(0).expect("bind device 0");
    let mut sink = CudaSink::new(0, 4096).expect("pinned staging");
    let source = tensor(4096); // 16 KiB, four times the staging budget
    let chunks = [&source[..]];
    let error = fill(
        &ChunkedSource::new(&chunks),
        &TensorFill {
            layout: arrangement("torch.contiguous@1").expect("record"),
            shape: vec![4096],
            element_bytes: 4,
            device_offset: 0,
        },
        &mut sink,
    )
    .expect_err("a tensor four times the staging budget was filled anyway");
    eprintln!("tensorfs#154: over-budget fill refused with: {error}");
}

// ---------------------------------------------------------------------------
// The gate's own proof. Runs on CPU, in every `cargo test`, needs no card.
// ---------------------------------------------------------------------------

#[test]
fn only_an_exact_1_on_a_sanctioning_variable_opens_the_gate() {
    let env = |pairs: &[(&str, &str)]| {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        move |key: &str| {
            owned
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        }
    };

    // Closed by default. This is the case that matters: a bare `cargo test` on
    // the shared box must not touch the card.
    assert!(!sanctioned(env(&[])), "an unset environment opened the gate");
    assert!(sanctioned(env(&[("TENSORFS_GPU_WINDOW", "1")])));
    assert!(sanctioned(env(&[("TENSORFS_GPU", "1")])));
    assert!(sanctioned(env(&[
        ("TENSORFS_GPU_WINDOW", "1"),
        ("TENSORFS_GPU", "1")
    ])));
    for bad in ["", "0", "true", "yes", "1 ", " 1", "01"] {
        assert!(
            !sanctioned(env(&[("TENSORFS_GPU_WINDOW", bad)])),
            "TENSORFS_GPU_WINDOW={bad:?} opened the gate"
        );
    }
}

/// A box with no driver gets a TYPED error, not a panic and not a hang. This is
/// the difference between "no card here" and "the fill path is broken", and it
/// runs everywhere — including on the card, where it proves the opposite branch.
#[test]
fn a_missing_driver_is_a_typed_error_and_a_present_one_loads() {
    match driver() {
        Ok(_) => eprintln!("libcuda.so.1 resolved: the device leg is reachable from this box"),
        Err(error) => {
            let rendered = error.to_string();
            assert!(
                rendered.contains("libcuda"),
                "a driver failure must name the driver: {rendered}"
            );
            eprintln!("no CUDA driver on this box, reported as: {rendered}");
        }
    }
}
