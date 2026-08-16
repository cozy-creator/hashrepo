use std::cell::RefCell;
use std::io;

use tensorfs_core::planner::{
    ByteSource, MAX_OBJECT_SIZE, Plan, PlanError, PlannerId, RegionKind, plan,
};

const MAX_HEADER_SIZE: u64 = 100_000_000;

fn try_plan<S: ByteSource + ?Sized>(source: &S) -> Result<Option<Plan>, PlanError> {
    let planned = plan(source)?;
    Ok((planned.planner() == PlannerId::SafetensorsV1).then_some(planned))
}

struct ShortSource(u64);

impl ByteSource for ShortSource {
    fn len(&self) -> u64 {
        self.0
    }

    fn read_exact_at(&self, _offset: u64, _destination: &mut [u8]) -> io::Result<()> {
        if _offset == 0 && _destination.len() == 4 {
            _destination.fill(0);
            return Ok(());
        }
        panic!("an impossible safetensors candidate must not be inspected beyond GGUF magic")
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

struct HeaderSource {
    header: Vec<u8>,
    data_size: u64,
    reads: RefCell<Vec<(u64, usize)>>,
    fail_at: Option<u64>,
}

impl HeaderSource {
    fn new(header: impl Into<Vec<u8>>, data_size: u64) -> Self {
        Self {
            header: header.into(),
            data_size,
            reads: RefCell::new(Vec::new()),
            fail_at: None,
        }
    }

    fn failing(header: impl Into<Vec<u8>>, data_size: u64, fail_at: u64) -> Self {
        Self {
            fail_at: Some(fail_at),
            ..Self::new(header, data_size)
        }
    }
}

impl ByteSource for HeaderSource {
    fn len(&self) -> u64 {
        8 + self.header.len() as u64 + self.data_size
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        self.reads.borrow_mut().push((offset, destination.len()));
        if self.fail_at == Some(offset) {
            return Err(io::Error::other("injected read failure"));
        }
        match offset {
            0 if destination.len() == 4 => {
                destination.copy_from_slice(&(self.header.len() as u64).to_le_bytes()[..4]);
                Ok(())
            }
            0 if destination.len() == 8 => {
                destination.copy_from_slice(&(self.header.len() as u64).to_le_bytes());
                Ok(())
            }
            8 if destination.len() == self.header.len() => {
                destination.copy_from_slice(&self.header);
                Ok(())
            }
            _ => Err(io::Error::other("planner read tensor body")),
        }
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

struct PaddedHeaderSource {
    header_size: u64,
    reads: RefCell<Vec<(u64, usize)>>,
}

impl ByteSource for PaddedHeaderSource {
    fn len(&self) -> u64 {
        8 + self.header_size
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        self.reads.borrow_mut().push((offset, destination.len()));
        if offset == 0 && destination.len() == 4 {
            destination.copy_from_slice(&self.header_size.to_le_bytes()[..4]);
            return Ok(());
        }
        if offset == 0 && destination.len() == 8 {
            destination.copy_from_slice(&self.header_size.to_le_bytes());
            return Ok(());
        }
        if offset == 8 && destination.len() as u64 == self.header_size {
            destination.fill(b' ');
            destination[..2].copy_from_slice(b"{}");
            return Ok(());
        }
        Err(io::Error::other("unexpected read"))
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

fn data_regions(header: &[u8], data_size: u64) -> Vec<(u64, u64, RegionKind)> {
    try_plan(&HeaderSource::new(header, data_size))
        .unwrap()
        .unwrap()
        .regions()
        .iter()
        .filter(|region| region.kind() == RegionKind::Tensor)
        .map(|region| {
            (
                region.offset() - 8 - header.len() as u64,
                region.length(),
                region.kind(),
            )
        })
        .collect()
}

fn region_rows(plan: &Plan) -> Vec<(u64, u64, RegionKind)> {
    plan.regions()
        .iter()
        .map(|region| (region.offset(), region.length(), region.kind()))
        .collect()
}

#[test]
fn exact_header_limit_is_valid_and_header_region_is_bounded() {
    let source = PaddedHeaderSource {
        header_size: MAX_HEADER_SIZE,
        reads: RefCell::new(Vec::new()),
    };

    let plan = try_plan(&source).unwrap().unwrap();

    assert_eq!(plan.planner(), PlannerId::SafetensorsV1);
    assert_eq!(
        region_rows(&plan),
        vec![
            (0, MAX_OBJECT_SIZE, RegionKind::Header),
            (
                MAX_OBJECT_SIZE,
                8 + MAX_HEADER_SIZE - MAX_OBJECT_SIZE,
                RegionKind::Header,
            ),
        ]
    );
    assert_eq!(
        *source.reads.borrow(),
        vec![(0, 4), (0, 8), (8, MAX_HEADER_SIZE as usize)]
    );
}

#[test]
fn header_above_upstream_limit_falls_back_without_reading_it() {
    let source = PaddedHeaderSource {
        header_size: MAX_HEADER_SIZE + 1,
        reads: RefCell::new(Vec::new()),
    };

    assert!(try_plan(&source).unwrap().is_none());
    assert_eq!(*source.reads.borrow(), vec![(0, 4), (0, 8)]);
}

#[test]
fn eight_and_nine_byte_candidates_fall_back_without_a_read() {
    for length in [8, 9] {
        let planned = plan(&ShortSource(length)).unwrap();
        assert_eq!(planned.planner(), PlannerId::RawFixed64mV1);
    }
}

#[test]
fn null_metadata_matches_the_released_and_upstream_parser() {
    let header = br#"{"__metadata__":null}"#;
    let planned = try_plan(&HeaderSource::new(header, 0)).unwrap().unwrap();

    assert_eq!(planned.planner(), PlannerId::SafetensorsV1);
}

#[test]
fn tensors_use_natural_and_tensor_relative_object_boundaries() {
    let exact = format!(
        r#"{{"x":{{"dtype":"U8","shape":[{MAX_OBJECT_SIZE}],"data_offsets":[0,{MAX_OBJECT_SIZE}]}}}}"#
    );
    assert_eq!(
        data_regions(exact.as_bytes(), MAX_OBJECT_SIZE),
        vec![(0, MAX_OBJECT_SIZE, RegionKind::Tensor)]
    );

    let larger = MAX_OBJECT_SIZE + 1;
    let split =
        format!(r#"{{"x":{{"dtype":"U8","shape":[{larger}],"data_offsets":[0,{larger}]}}}}"#);
    assert_eq!(
        data_regions(split.as_bytes(), larger),
        vec![
            (0, MAX_OBJECT_SIZE, RegionKind::Tensor),
            (MAX_OBJECT_SIZE, 1, RegionKind::Tensor),
        ]
    );
}

#[test]
fn declaration_reordering_does_not_change_the_physical_range_plan() {
    let first = br#"{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"U8","shape":[5],"data_offsets":[3,8]}}"#;
    let reordered = br#"{"b":{"dtype":"U8","shape":[5],"data_offsets":[3,8]},"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#;

    let first_plan = try_plan(&HeaderSource::new(first, 8)).unwrap().unwrap();
    let reordered_plan = try_plan(&HeaderSource::new(reordered, 8)).unwrap().unwrap();

    assert_eq!(first.len(), reordered.len());
    assert_eq!(first_plan, reordered_plan);
}

#[test]
fn inserting_a_small_tensor_does_not_pack_neighboring_tensors_together() {
    let before = br#"{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"U8","shape":[5],"data_offsets":[3,8]}}"#;
    let after = br#"{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"new":{"dtype":"U8","shape":[2],"data_offsets":[3,5]},"b":{"dtype":"U8","shape":[5],"data_offsets":[5,10]}}"#;

    assert_eq!(
        data_regions(before, 8)
            .iter()
            .map(|region| region.1)
            .collect::<Vec<_>>(),
        vec![3, 5]
    );
    assert_eq!(
        data_regions(after, 10)
            .iter()
            .map(|region| region.1)
            .collect::<Vec<_>>(),
        vec![3, 2, 5]
    );
}

#[test]
fn zero_length_tensors_create_no_data_objects() {
    let header =
        br#"{"empty":{"dtype":"F32","shape":[0,18446744073709551615],"data_offsets":[0,0]}}"#;

    let plan = try_plan(&HeaderSource::new(header, 0)).unwrap().unwrap();

    assert_eq!(
        region_rows(&plan),
        vec![(0, 8 + header.len() as u64, RegionKind::Header,)]
    );
}

#[test]
fn every_known_dtype_has_strict_byte_geometry() {
    let cases = [
        ("F4", 2, 1),
        ("F6_E2M3", 4, 3),
        ("F6_E3M2", 4, 3),
        ("BOOL", 1, 1),
        ("U8", 1, 1),
        ("I8", 1, 1),
        ("F8_E5M2", 1, 1),
        ("F8_E4M3", 1, 1),
        ("F8_E8M0", 1, 1),
        ("F8_E5M2FNUZ", 1, 1),
        ("F8_E4M3FNUZ", 1, 1),
        ("I16", 1, 2),
        ("U16", 1, 2),
        ("F16", 1, 2),
        ("BF16", 1, 2),
        ("I32", 1, 4),
        ("U32", 1, 4),
        ("F32", 1, 4),
        ("I64", 1, 8),
        ("U64", 1, 8),
        ("F64", 1, 8),
        ("C64", 1, 8),
    ];

    for (dtype, elements, bytes) in cases {
        let header = format!(
            r#"{{"x":{{"dtype":"{dtype}","shape":[{elements}],"data_offsets":[0,{bytes}]}}}}"#
        );
        assert!(
            try_plan(&HeaderSource::new(header, bytes))
                .unwrap()
                .is_some(),
            "dtype {dtype} should be recognized"
        );
    }
}

#[test]
fn malformed_headers_fall_back_as_a_whole() {
    let malformed: &[(&[u8], u64)] = &[
        (b"not-json", 0),
        (b"[]", 0),
        (
            br#"{"x":{"dtype":"F128","shape":[1],"data_offsets":[0,16]}}"#,
            16,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}"#,
            1,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[18446744073709551615,2,0],"data_offsets":[0,0]}}"#,
            0,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[18446744073709551616],"data_offsets":[0,0]}}"#,
            0,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            2,
        ),
        (
            br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
            3,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            2,
        ),
        (
            br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":true}}"#,
            1,
        ),
        (br#"{"__metadata__":{"bad":1}}"#, 0),
        (br#"{"": {"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#, 0),
    ];

    for (header, data_size) in malformed {
        assert!(
            try_plan(&HeaderSource::new(*header, *data_size))
                .unwrap()
                .is_none(),
            "malformed header should fall back: {}",
            String::from_utf8_lossy(header)
        );
    }
}

#[test]
fn duplicate_keys_at_every_object_level_fall_back() {
    let duplicate_headers: &[&[u8]] = &[
        br#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
        br#"{"x":{"dtype":"U8","dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
        br#"{"__metadata__":{"key":"one","key":"two"}}"#,
    ];

    for header in duplicate_headers {
        assert!(try_plan(&HeaderSource::new(*header, 0)).unwrap().is_none());
    }
}

#[test]
fn only_structurally_requested_reads_can_surface_io_errors() {
    let valid = br#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
    let prefix_failure = HeaderSource::failing(valid, 0, 0);
    assert!(matches!(try_plan(&prefix_failure), Err(PlanError::Read(_))));

    let header_failure = HeaderSource::failing(valid, 0, 8);
    assert!(matches!(try_plan(&header_failure), Err(PlanError::Read(_))));

    let malformed = HeaderSource::new(b"not-json".as_slice(), 1_000_000_000);
    assert!(try_plan(&malformed).unwrap().is_none());
    assert_eq!(
        *malformed.reads.borrow(),
        vec![(0, 4), (0, 8), (8, b"not-json".len())]
    );
}

#[test]
fn the_minimal_two_byte_unpadded_header_is_valid_safetensors() {
    let header = br#"{}"#;
    let plan = try_plan(&HeaderSource::new(header, 0)).unwrap().unwrap();

    assert_eq!(plan.planner(), PlannerId::SafetensorsV1);
    assert_eq!(region_rows(&plan), vec![(0, 10, RegionKind::Header)]);
}

#[test]
fn sub_byte_dtypes_fall_back_unless_the_element_count_is_byte_aligned() {
    // 4- and 6-bit elements only have byte geometry at whole-byte multiples.
    // A misaligned count is a structural refusal, not a rounded object length.
    //
    // Every row below declares the byte length that a *truncating* bit-to-byte
    // conversion would produce, so the only thing standing between these
    // headers and a plan is the alignment guard itself.
    let misaligned: &[(&[u8], u64)] = &[
        (
            br#"{"x":{"dtype":"F4","shape":[1],"data_offsets":[0,0]}}"#,
            0,
        ),
        (
            br#"{"x":{"dtype":"F4","shape":[3],"data_offsets":[0,1]}}"#,
            1,
        ),
        (
            br#"{"x":{"dtype":"F6_E2M3","shape":[1],"data_offsets":[0,0]}}"#,
            0,
        ),
        (
            br#"{"x":{"dtype":"F6_E3M2","shape":[2],"data_offsets":[0,1]}}"#,
            1,
        ),
    ];
    for (header, data_size) in misaligned {
        assert!(
            try_plan(&HeaderSource::new(*header, *data_size))
                .unwrap()
                .is_none(),
            "misaligned sub-byte tensor should fall back: {}",
            String::from_utf8_lossy(header)
        );
    }

    let aligned: &[(&[u8], u64)] = &[
        (
            br#"{"x":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#,
            1,
        ),
        (
            br#"{"x":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#,
            2,
        ),
        (
            br#"{"x":{"dtype":"F6_E2M3","shape":[4],"data_offsets":[0,3]}}"#,
            3,
        ),
        (
            br#"{"x":{"dtype":"F6_E3M2","shape":[4],"data_offsets":[0,3]}}"#,
            3,
        ),
    ];
    for (header, data_size) in aligned {
        assert!(
            try_plan(&HeaderSource::new(*header, *data_size))
                .unwrap()
                .is_some(),
            "byte-aligned sub-byte tensor should plan: {}",
            String::from_utf8_lossy(header)
        );
    }
}

#[test]
fn a_pathologically_nested_header_falls_back_instead_of_escaping_the_parser() {
    const DEPTH: usize = 10_000;
    let mut header = Vec::from(br#"{"x":{"dtype":"U8","shape":"#.as_slice());
    header.extend(std::iter::repeat_n(b'[', DEPTH));
    header.push(b'0');
    header.extend(std::iter::repeat_n(b']', DEPTH));
    header.extend_from_slice(br#","data_offsets":[0,0]}}"#);

    // The refusal must be an ordinary whole-file fallback that returns cleanly.
    // A parser without a depth bound would abort the process here instead.
    let source = HeaderSource::new(header, 0);
    assert!(try_plan(&source).unwrap().is_none());
    assert_eq!(plan(&source).unwrap().planner(), PlannerId::RawFixed64mV1);

    // Contrast: the identical header with an unnested shape plans normally, so
    // the fallback above is caused by the nesting and not by the surrounding
    // fixture.
    let flat = br#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
    assert!(try_plan(&HeaderSource::new(flat, 0)).unwrap().is_some());
}

#[test]
fn an_enormous_shape_declaration_is_refused_before_any_shape_arithmetic() {
    // 2,000 dimensions of 1,000 digits each. The refusal is structural rather
    // than timed: `shape` decodes as `u64`, so the first out-of-range dimension
    // ends the parse without the planner ever folding a product.
    let dimension = "9".repeat(1_000);
    let shape = vec![dimension.as_str(); 2_000].join(",");
    let header =
        format!(r#"{{"x":{{"dtype":"U8","shape":[{shape}],"data_offsets":[0,1]}}}}"#).into_bytes();

    let source = HeaderSource::new(header, 1);
    assert!(try_plan(&source).unwrap().is_none());
    assert_eq!(plan(&source).unwrap().planner(), PlannerId::RawFixed64mV1);

    // Contrast: the same 2,000-dimension shape made of in-range dimensions
    // whose product still matches the declared bytes plans normally, so the
    // refusal above is the out-of-range dimension rather than the arity.
    let ones = vec!["1"; 2_000].join(",");
    let valid =
        format!(r#"{{"x":{{"dtype":"U8","shape":[{ones}],"data_offsets":[0,1]}}}}"#).into_bytes();
    assert!(try_plan(&HeaderSource::new(valid, 1)).unwrap().is_some());
}
