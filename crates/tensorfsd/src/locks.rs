//! POSIX advisory record locks for one mount.
//!
//! The table is per-mount by construction and dies with it: advisory locks
//! are session state, never workspace facts, so nothing here persists or
//! enters a snapshot. Ranges use the FUSE wire shape — inclusive
//! `[start, end]` byte offsets, with `end == u64::MAX` standing for "to end
//! of file" (`l_len == 0`).
//!
//! Semantics follow POSIX record locking: shared locks are compatible with
//! shared locks, exclusive locks conflict with everything held by another
//! owner, and an owner's new lock replaces its own locks over the overlapping
//! range (splitting partial overlaps). Closing any descriptor releases every
//! lock its owner holds on that file, which the kernel reports through
//! `FLUSH`/`RELEASE` with the owning `lock_owner`.

use std::collections::HashMap;

/// One granted advisory lock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PosixLock {
    pub owner: u64,
    pub pid: u32,
    pub exclusive: bool,
    /// Inclusive first byte.
    pub start: u64,
    /// Inclusive last byte; `u64::MAX` means to end of file.
    pub end: u64,
}

impl PosixLock {
    fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start <= end && start <= self.end
    }

    fn conflicts_with(&self, owner: u64, exclusive: bool, start: u64, end: u64) -> bool {
        self.owner != owner && (self.exclusive || exclusive) && self.overlaps(start, end)
    }
}

/// Per-mount advisory lock state, keyed by inode.
#[derive(Default)]
pub(crate) struct LockTable {
    by_ino: HashMap<u64, Vec<PosixLock>>,
}

impl LockTable {
    /// The first lock by another owner that blocks the described request, in
    /// `getlk` reporting order (lowest start first).
    pub fn first_conflict(
        &self,
        ino: u64,
        owner: u64,
        exclusive: bool,
        start: u64,
        end: u64,
    ) -> Option<PosixLock> {
        let mut conflicts: Vec<&PosixLock> = self
            .by_ino
            .get(&ino)
            .into_iter()
            .flatten()
            .filter(|lock| lock.conflicts_with(owner, exclusive, start, end))
            .collect();
        conflicts.sort_by_key(|lock| lock.start);
        conflicts.first().map(|lock| **lock)
    }

    /// Grants the lock or reports the first conflicting holder. A grant
    /// replaces the owner's own locks over the range, POSIX-style.
    pub fn set(
        &mut self,
        ino: u64,
        owner: u64,
        pid: u32,
        exclusive: bool,
        start: u64,
        end: u64,
    ) -> Result<(), PosixLock> {
        if let Some(conflict) = self.first_conflict(ino, owner, exclusive, start, end) {
            return Err(conflict);
        }
        let locks = self.by_ino.entry(ino).or_default();
        carve(locks, owner, start, end);
        locks.push(PosixLock {
            owner,
            pid,
            exclusive,
            start,
            end,
        });
        merge_owner(locks, owner);
        Ok(())
    }

    /// Removes the owner's coverage of `[start, end]`, splitting partial
    /// overlaps. Unlocking a range nobody holds is a successful no-op.
    pub fn unset(&mut self, ino: u64, owner: u64, start: u64, end: u64) {
        if let Some(locks) = self.by_ino.get_mut(&ino) {
            carve(locks, owner, start, end);
            if locks.is_empty() {
                self.by_ino.remove(&ino);
            }
        }
    }

    /// Releases every lock the owner holds on the inode: POSIX close-releases
    /// semantics, driven by `FLUSH`/`RELEASE`.
    pub fn release_owner(&mut self, ino: u64, owner: u64) {
        if let Some(locks) = self.by_ino.get_mut(&ino) {
            locks.retain(|lock| lock.owner != owner);
            if locks.is_empty() {
                self.by_ino.remove(&ino);
            }
        }
    }

    /// Drops every lock on an inode that left the tree.
    pub fn forget_ino(&mut self, ino: u64) {
        self.by_ino.remove(&ino);
    }
}

/// Removes `owner`'s coverage of the inclusive `[start, end]` range,
/// re-inserting the fragments of partially covered locks.
fn carve(locks: &mut Vec<PosixLock>, owner: u64, start: u64, end: u64) {
    let mut fragments = Vec::new();
    locks.retain(|lock| {
        if lock.owner != owner || !lock.overlaps(start, end) {
            return true;
        }
        if lock.start < start {
            fragments.push(PosixLock {
                end: start - 1,
                ..*lock
            });
        }
        if lock.end > end {
            fragments.push(PosixLock {
                start: end + 1,
                ..*lock
            });
        }
        false
    });
    locks.extend(fragments);
}

/// Coalesces the owner's adjacent same-type ranges so the table stays
/// canonical and `getlk` reports whole extents.
fn merge_owner(locks: &mut Vec<PosixLock>, owner: u64) {
    let mut mine: Vec<PosixLock> = Vec::new();
    locks.retain(|lock| {
        if lock.owner == owner {
            mine.push(*lock);
            false
        } else {
            true
        }
    });
    mine.sort_by_key(|lock| lock.start);
    let mut merged: Vec<PosixLock> = Vec::new();
    for lock in mine {
        match merged.last_mut() {
            Some(last)
                if last.exclusive == lock.exclusive
                    && last.end != u64::MAX
                    && last.end + 1 >= lock.start =>
            {
                last.end = last.end.max(lock.end);
            }
            _ => merged.push(lock),
        }
    }
    locks.extend(merged);
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 11;
    const B: u64 = 22;

    #[test]
    fn shared_locks_share_and_exclusive_locks_do_not() {
        let mut table = LockTable::default();
        table.set(1, A, 100, false, 0, 99).expect("shared grants");
        table.set(1, B, 200, false, 50, 149).expect("shared shares");
        let conflict = table
            .set(1, B, 200, true, 60, 70)
            .expect_err("exclusive over another owner's shared range refuses");
        assert_eq!((conflict.owner, conflict.pid), (A, 100));
        // Upgrading over a range that overlaps ANOTHER owner refuses; a range
        // covered only by the requester's own shared lock upgrades freely.
        assert!(table.set(1, B, 200, true, 90, 149).is_err());
        table.set(1, B, 200, true, 100, 149).expect("self-upgrade");
        table
            .set(1, B, 200, true, 150, 250)
            .expect("disjoint grants");
    }

    #[test]
    fn an_owner_replaces_and_splits_its_own_ranges() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 0, 9_999).expect("grants");
        // Downgrade of a middle slice replaces in place, no self-conflict.
        table.set(1, A, 100, false, 4_000, 5_999).expect("replaces");
        table.unset(1, A, 4_000, 5_999);
        // The carved middle is free for another owner; the edges are not.
        table
            .set(1, B, 200, true, 4_000, 5_999)
            .expect("middle freed");
        assert!(table.set(1, B, 200, true, 0, 100).is_err());
        assert!(table.set(1, B, 200, true, 6_000, 6_100).is_err());
    }

    #[test]
    fn release_owner_frees_everything_it_held() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 0, u64::MAX).expect("grants");
        assert!(table.set(1, B, 200, false, 5, 6).is_err());
        table.release_owner(1, A);
        table.set(1, B, 200, true, 0, u64::MAX).expect("freed");
    }

    #[test]
    fn getlk_reports_the_lowest_conflicting_extent() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 500, 599).expect("grants");
        table.set(1, A, 100, true, 100, 199).expect("grants");
        let hit = table
            .first_conflict(1, B, false, 0, u64::MAX)
            .expect("conflicts");
        assert_eq!((hit.start, hit.end, hit.pid), (100, 199, 100));
        assert!(table.first_conflict(1, A, true, 0, u64::MAX).is_none());
    }

    #[test]
    fn adjacent_same_type_ranges_coalesce() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 0, 99).expect("grants");
        table.set(1, A, 100, true, 100, 199).expect("grants");
        let extent = table
            .first_conflict(1, B, false, 150, 150)
            .expect("conflicts");
        assert_eq!((extent.start, extent.end), (0, 199));
        // A to-EOF lock never wraps past u64::MAX while merging.
        table.set(1, A, 100, true, 200, u64::MAX).expect("grants");
        let extent = table
            .first_conflict(1, B, false, 300, 300)
            .expect("conflicts");
        assert_eq!(extent.end, u64::MAX);
    }

    /// Ranges are inclusive on BOTH ends, so a request touching a held lock at
    /// exactly one byte conflicts and one byte past it does not.
    ///
    /// Every other test here works in ranges wide enough that an off-by-one in
    /// `overlaps` is invisible: turning either `<=` into `<` left the whole
    /// suite green. Each direction is pinned separately, because the two
    /// comparisons guard opposite edges.
    #[test]
    fn overlap_is_inclusive_at_both_edges_by_exactly_one_byte() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 10, 20).expect("grants");

        // Upper edge: `self.start <= end`.
        assert!(
            table.first_conflict(1, B, true, 0, 10).is_some(),
            "a request ending ON the held lock's first byte conflicts"
        );
        assert!(
            table.first_conflict(1, B, true, 0, 9).is_none(),
            "a request ending one byte BEFORE it does not"
        );

        // Lower edge: `start <= self.end`.
        assert!(
            table.first_conflict(1, B, true, 20, 30).is_some(),
            "a request starting ON the held lock's last byte conflicts"
        );
        assert!(
            table.first_conflict(1, B, true, 21, 30).is_none(),
            "a request starting one byte AFTER it does not"
        );
    }

    /// `getlk` must report the LOWEST conflicting extent, whichever holder
    /// happens to be stored first.
    ///
    /// The existing coverage only ever conflicts against one owner, and
    /// `merge_owner` keeps a single owner's locks sorted incidentally — so
    /// `first_conflict`'s own sort was doing nothing observable and deleting
    /// it changed no result. Two owners, inserted in reverse start order, is
    /// the arrangement no incidental ordering can normalise.
    #[test]
    fn getlk_reports_the_lowest_start_across_holders_not_insertion_order() {
        const C: u64 = 33;
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 500, 599).expect("grants");
        table.set(1, B, 200, true, 100, 199).expect("grants");

        let hit = table
            .first_conflict(1, C, true, 0, u64::MAX)
            .expect("conflicts");
        assert_eq!(
            (hit.start, hit.pid),
            (100, 200),
            "getlk reports the lowest conflicting extent, not the one inserted first"
        );
    }

    /// An inode that leaves the tree takes its locks with it.
    ///
    /// `forget_ino` had no test at all — making it a no-op left the suite
    /// green — so a forgotten inode could have kept refusing locks forever,
    /// and a recycled inode number would have inherited a stranger's locks.
    #[test]
    fn forgetting_an_inode_drops_every_lock_held_on_it() {
        let mut table = LockTable::default();
        table.set(1, A, 100, true, 0, 99).expect("grants");
        table.set(2, A, 100, true, 0, 99).expect("grants");
        assert!(table.first_conflict(1, B, true, 0, 99).is_some());

        table.forget_ino(1);

        assert!(
            table.first_conflict(1, B, true, 0, 99).is_none(),
            "a forgotten inode holds no locks"
        );
        table
            .set(1, B, 200, true, 0, 99)
            .expect("and another owner may now take the range");
        assert!(
            table.first_conflict(2, B, true, 0, 99).is_some(),
            "forgetting one inode must not disturb another"
        );
    }
}
