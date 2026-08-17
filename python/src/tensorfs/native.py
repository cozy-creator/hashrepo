"""The compiled half of the `tensorfs` distribution.

`tensorfs` is one distribution containing a pure-Python facade and a compiled
Rust extension, the way `huggingface_hub` and `hf_xet` are related -- except
that here they are the same wheel rather than two.

This module is the public name for the extension. `tensorfs._tensorfs` is the
extension itself and is private; import from here.

Nothing in the existing pure-Python modules has been rewritten onto Rust.
`LocalCAS`, `transfer` and the rest are unchanged; this is the binding they
will be ported onto later.

There is no `tensorfsd` mount daemon in the wheel or on this branch. Pods
cannot mount FUSE -- opening `/dev/fuse` is denied by the device cgroup even
for root, `CAP_SYS_ADMIN` is absent from the container's bounding set, and
there is no API field to grant either -- so the wheel carries native reads,
and the daemon is shelved on the `shelf/tensorfsd` branch (tag
`shelf/tensorfsd-split`).
"""

from __future__ import annotations

from ._tensorfs import (
    MAX_FILE_RECORDS,
    MAX_OBJECT_SIZE,
    MAX_PACK_OBJECTS,
    MAX_PACK_PAYLOAD,
    MAX_PATH_BYTES,
    MAX_SYMLINK_TARGET_BYTES,
    STUB_MAGIC,
    AdmittedObject,
    ContractRegistry,
    FileRecord,
    FormatError,
    HashedPlan,
    LayoutError,
    MappedObject,
    ObjectStore,
    PackObject,
    Plan,
    PlanError,
    PlannedObject,
    RecordsReader,
    Region,
    Snapshot,
    SnapshotEntry,
    StoreError,
    TempCollection,
    TensorfsError,
    __version__,
    adopt,
    decode_pack,
    decode_snapshot,
    derive,
    encode_pack,
    ingest_concurrency,
    parse_stub,
    plan_and_hash_bytes,
    plan_and_hash_file,
    plan_bytes,
    plan_file,
    rekey,
    snapshot_id_of,
    stub_bytes,
    subset,
)

__all__ = [
    "MAX_FILE_RECORDS",
    "MAX_OBJECT_SIZE",
    "MAX_PACK_OBJECTS",
    "MAX_PACK_PAYLOAD",
    "MAX_PATH_BYTES",
    "MAX_SYMLINK_TARGET_BYTES",
    "STUB_MAGIC",
    "AdmittedObject",
    "ContractRegistry",
    "FileRecord",
    "FormatError",
    "HashedPlan",
    "LayoutError",
    "MappedObject",
    "ObjectStore",
    "PackObject",
    "Plan",
    "PlanError",
    "PlannedObject",
    "RecordsReader",
    "Region",
    "Snapshot",
    "SnapshotEntry",
    "StoreError",
    "TempCollection",
    "TensorfsError",
    "__version__",
    "adopt",
    "decode_pack",
    "decode_snapshot",
    "derive",
    "encode_pack",
    "ingest_concurrency",
    "plan_and_hash_bytes",
    "plan_and_hash_file",
    "parse_stub",
    "plan_bytes",
    "plan_file",
    "rekey",
    "snapshot_id_of",
    "stub_bytes",
    "subset",
]
