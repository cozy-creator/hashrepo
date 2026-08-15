use sha2::{Digest, Sha256};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::MAX_OBJECT_SIZE;
use tensorfs_core::tfp1::{MAX_PACK_OBJECTS, MAX_PACK_PAYLOAD, Pack, Tfp1Error, decode, encode};

fn digest_of(bytes: &[u8]) -> ObjectDigest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    ObjectDigest::from_bytes(hasher.finalize().into())
}

fn rows<'a>(objects: &[&'a [u8]]) -> Vec<(ObjectDigest, &'a [u8])> {
    objects
        .iter()
        .map(|bytes| (digest_of(bytes), *bytes))
        .collect()
}

#[test]
fn encoding_is_canonical_and_round_trips_byte_exactly() {
    let a = b"attention-weights".as_slice();
    let b = b"adaln-weights".as_slice();
    let c = b"x".as_slice();

    let forward = encode(&rows(&[a, b, c])).unwrap();
    let shuffled = encode(&rows(&[c, a, b])).unwrap();
    assert_eq!(
        forward, shuffled,
        "insertion order must not reach the canonical bytes"
    );

    let pack = decode(&forward).unwrap();
    assert_eq!(pack.object_count(), 3);

    let mut digests: Vec<ObjectDigest> = pack.objects().map(|object| object.digest()).collect();
    let decoded: Vec<(ObjectDigest, &[u8])> = pack
        .objects()
        .map(|object| (object.digest(), object.bytes()))
        .collect();
    assert_eq!(encode(&decoded).unwrap(), forward, "re-encode drifted");

    let sorted = {
        digests.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        digests
    };
    assert_eq!(
        pack.objects()
            .map(|object| object.digest())
            .collect::<Vec<_>>(),
        sorted,
        "index order must be ascending digest order"
    );
}

#[test]
fn payload_offsets_tile_the_pack_in_index_order() {
    let bytes = encode(&rows(&[b"first", b"second", b"third"])).unwrap();
    let pack = Pack::parse(&bytes).unwrap();

    let mut concatenated = Vec::new();
    for object in pack.objects() {
        concatenated.extend_from_slice(object.bytes());
    }
    assert_eq!(
        &bytes[bytes.len() - concatenated.len()..],
        &concatenated[..],
        "objects must tile the payload region exactly, in index order"
    );
}

#[test]
fn the_builder_refuses_every_bounds_and_identity_violation() {
    assert!(matches!(encode(&[]), Err(Tfp1Error::ZeroObjects)));

    let object = b"payload".as_slice();
    assert!(matches!(
        encode(&[(digest_of(object), b"".as_slice())]),
        Err(Tfp1Error::ZeroLengthObject)
    ));
    assert!(matches!(
        encode(&[(digest_of(b"other"), object)]),
        Err(Tfp1Error::DigestMismatch)
    ));
    assert!(matches!(
        encode(&rows(&[object, object])),
        Err(Tfp1Error::DuplicateDigest)
    ));

    let too_many: Vec<(ObjectDigest, &[u8])> =
        vec![(digest_of(object), object); MAX_PACK_OBJECTS + 1];
    assert!(matches!(encode(&too_many), Err(Tfp1Error::ObjectLimit)));
}

#[test]
fn one_maximum_size_object_fills_a_pack_and_one_more_byte_refuses() {
    assert_eq!(MAX_PACK_PAYLOAD, MAX_OBJECT_SIZE);
    let full = vec![0xa5_u8; MAX_OBJECT_SIZE as usize];

    let bytes = encode(&rows(&[&full])).unwrap();
    let pack = decode(&bytes).unwrap();
    assert_eq!(pack.object_count(), 1);
    assert_eq!(
        pack.objects().next().unwrap().bytes().len() as u64,
        MAX_OBJECT_SIZE
    );

    // A second object cannot fit beside a full one.
    assert!(matches!(
        encode(&rows(&[&full, b"x"])),
        Err(Tfp1Error::PayloadTooLarge)
    ));

    // One object above the object bound refuses on its own.
    let oversize = vec![0x5a_u8; MAX_OBJECT_SIZE as usize + 1];
    assert!(matches!(
        encode(&rows(&[&oversize])),
        Err(Tfp1Error::ObjectTooLarge)
    ));
}

#[test]
fn structural_parse_accepts_what_digest_verification_then_refuses() {
    let mut bytes = encode(&rows(&[b"stream-verified bytes"])).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;

    let pack = Pack::parse(&bytes).expect("the mutation is structurally silent");
    assert!(matches!(
        pack.verify_objects(),
        Err(Tfp1Error::DigestMismatch)
    ));
    assert!(matches!(decode(&bytes), Err(Tfp1Error::DigestMismatch)));
}

/// TFP1 is transport staging only. There is no API from pack bytes to any
/// snapshot or object identity: `Pack` exposes only the indexed objects, and
/// nothing in `tfm1` or `object` accepts an envelope. This arm pins the
/// weaker, testable half — the pack's own hash never collides into the
/// identity vocabulary it carries.
#[test]
fn pack_bytes_never_carry_an_identity_of_their_own() {
    let objects = rows(&[b"one".as_slice(), b"two".as_slice()]);
    let bytes = encode(&objects).unwrap();

    let pack_hash = digest_of(&bytes);
    for (digest, _) in &objects {
        assert_ne!(pack_hash, *digest);
    }
}
