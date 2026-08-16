# tensorfsd is shelved here

`shelf/tensorfsd` is the permanent reference copy of the FUSE daemon. The
fork point is tagged `shelf/tensorfsd-split` — the pre-squash `master` tip.
`master` itself was squashed to a single genesis root on 2026-08-16, so this
branch and that tag are also the repository's reachable pre-squash history.
What lives here and was deleted from `master`:

- `crates/tensorfsd/` — the mount daemon, control plane, the six Linux
  mount/RPC/lock test suites, and the `tensorfs-bench` measurement harness;
- `python/src/tensorfs/daemon.py` and `python/tests/test_daemon.py` — the
  daemon client and its tests;
- `scripts/check-daemon-linux-gate.sh` and its CI step;
- `benchmarks/` — the pod FUSE-mount benchmark campaign (`tensorfs-bench`
  docs, `run_on_pod.sh`, mount read/write arms, RESULTS).

Shelved 2026-08-16, PR "shelve tensorfsd: the daemon moves to
shelf/tensorfsd". Why: pods cannot mount FUSE at all (`/dev/fuse` denied by
the device cgroup, `CAP_SYS_ADMIN` absent), direct tensor reads/writes
replaced the mount as the deployment path (`docs/direct-tensor-reads.md`),
and the daemon was receiving no further investment — keeping it in
`master`'s working set cost CI time and attention for code nobody was
developing.

Rules for this branch:

- **It is never rebased.** It stays buildable at its own commit, against the
  `tensorfs-core` it was written for. Do not try to keep it current.
- **Never delete this branch or its tag.** Beyond the daemon, they are the
  only remaining anchor of the pre-squash history the genesis commit points
  at.
- **Revival is rewrite-with-reference**, not a merge: port the ideas against
  whatever core looks like then, using this tree as the reference. Issues
  #59 (macFUSE/FSKit and WinFsp adapters) and #50 (`snapshot_fs` reports the
  wrong inode for `..` — shipped here unfixed, fix on revival) are the
  starting points.
