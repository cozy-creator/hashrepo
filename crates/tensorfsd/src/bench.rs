//! Row schema and folds for the pgw#1256 measured matrix.
//!
//! The `tensorfs-bench` bin emits one JSONL stream per run: a single `meta`
//! row carrying machine/filesystem/package provenance, then one `arm` row per
//! (arm, repetition), then one `summary` row per arm folding wall-clock
//! distributions. Nothing here asserts timing: rows are evidence, and the
//! release gate reads them off a quiet host.

use serde::{Deserialize, Serialize};

/// The JSONL row schema version; pre-launch, it is replaced in place.
pub const BENCH_SCHEMA: u32 = 1;

/// Run-level provenance. Every field is required: a row set that cannot say
/// what produced it is not evidence.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetaRow {
    pub schema: u32,
    pub kind: String,
    pub run_id: String,
    pub scale_bytes: u64,
    pub reps: u32,
    pub kernel: String,
    pub fs_type: String,
    /// SHA-256 prefix of the hostname: comparable across runs, no name leak.
    pub machine_hash: String,
    pub crate_version: String,
    pub started_unix: u64,
}

/// One measured arm repetition. Optional fields are absent when an arm has
/// nothing honest to report for them, never zero-filled.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArmRow {
    pub schema: u32,
    pub kind: String,
    pub run_id: String,
    pub arm: String,
    pub rep: u32,
    pub wall_s: f64,
    pub user_s: f64,
    pub sys_s: f64,
    /// Process-lifetime high-water mark at the end of the arm (VmHWM); a
    /// per-arm peak is not observable without instrumentation.
    pub peak_rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    /// fsync/fdatasync calls the driver itself issued through the mount.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_fsyncs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects_new: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects_reused: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_new: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_reused: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store_objects_after: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store_physical_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_bytes: Option<u64>,
    /// 1-minute load average when the arm started.
    pub load_1m: f64,
    /// True when `load_1m` exceeded 4.0: the row self-describes its noise.
    pub load_caveat: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ArmRow {
    /// The accounting identity every reuse-reporting row must satisfy.
    #[must_use]
    pub fn accounting_consistent(&self) -> bool {
        match (self.bytes_new, self.bytes_reused, self.bytes_total) {
            (Some(new), Some(reused), Some(total)) => {
                new.checked_add(reused).is_some_and(|sum| sum == total)
            }
            _ => true,
        }
    }
}

/// Per-arm wall-clock distribution over repetitions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryRow {
    pub schema: u32,
    pub kind: String,
    pub run_id: String,
    pub arm: String,
    pub reps: u32,
    pub wall_min_s: f64,
    pub wall_median_s: f64,
    pub wall_max_s: f64,
}

/// Folds one arm's repetitions into its wall-clock distribution.
#[must_use]
pub fn summarize(run_id: &str, arm: &str, rows: &[ArmRow]) -> Option<SummaryRow> {
    let mut walls: Vec<f64> = rows
        .iter()
        .filter(|row| row.arm == arm)
        .map(|row| row.wall_s)
        .collect();
    if walls.is_empty() {
        return None;
    }
    walls.sort_by(|left, right| left.total_cmp(right));
    Some(SummaryRow {
        schema: BENCH_SCHEMA,
        kind: "summary".to_owned(),
        run_id: run_id.to_owned(),
        arm: arm.to_owned(),
        reps: walls.len() as u32,
        wall_min_s: walls[0],
        wall_median_s: walls[walls.len() / 2],
        wall_max_s: walls[walls.len() - 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(arm: &str, rep: u32, wall: f64) -> ArmRow {
        ArmRow {
            schema: BENCH_SCHEMA,
            kind: "arm".to_owned(),
            run_id: "run".to_owned(),
            arm: arm.to_owned(),
            rep,
            wall_s: wall,
            user_s: 0.1,
            sys_s: 0.1,
            peak_rss_bytes: 1,
            io_read_bytes: 2,
            io_write_bytes: 3,
            driver_fsyncs: None,
            objects_new: Some(4),
            objects_reused: Some(6),
            bytes_new: Some(40),
            bytes_reused: Some(60),
            bytes_total: Some(100),
            store_objects_after: None,
            store_physical_bytes: None,
            logical_bytes: None,
            load_1m: 1.0,
            load_caveat: false,
            note: None,
        }
    }

    #[test]
    fn arm_rows_round_trip_and_reject_unknown_fields() {
        let original = row("import", 0, 2.5);
        let encoded = serde_json::to_string(&original).expect("row serializes");
        let decoded: ArmRow = serde_json::from_str(&encoded).expect("row parses");
        assert_eq!(original, decoded);

        let sneaky = encoded.replace("\"load_caveat\":false", "\"load_caveat\":false,\"x\":1");
        assert!(serde_json::from_str::<ArmRow>(&sneaky).is_err());
    }

    #[test]
    fn meta_rows_require_every_provenance_field() {
        let meta = MetaRow {
            schema: BENCH_SCHEMA,
            kind: "meta".to_owned(),
            run_id: "run".to_owned(),
            scale_bytes: 1,
            reps: 1,
            kernel: "k".to_owned(),
            fs_type: "ext4".to_owned(),
            machine_hash: "ab12".to_owned(),
            crate_version: "0.1.0".to_owned(),
            started_unix: 1,
        };
        let encoded = serde_json::to_string(&meta).expect("meta serializes");
        assert_eq!(
            meta,
            serde_json::from_str::<MetaRow>(&encoded).expect("parses")
        );

        // A meta row that cannot say what produced it must refuse to parse.
        let anonymous = encoded.replace("\"machine_hash\":\"ab12\",", "");
        assert!(serde_json::from_str::<MetaRow>(&anonymous).is_err());
        let versionless = encoded.replace("\"crate_version\":\"0.1.0\",", "");
        assert!(serde_json::from_str::<MetaRow>(&versionless).is_err());
    }

    #[test]
    fn the_reuse_accounting_identity_binds() {
        let mut consistent = row("reseal", 0, 1.0);
        assert!(consistent.accounting_consistent());

        consistent.bytes_reused = Some(61);
        assert!(!consistent.accounting_consistent());

        // Overflow must read as inconsistent, never wrap to a match.
        consistent.bytes_new = Some(u64::MAX);
        consistent.bytes_reused = Some(1);
        consistent.bytes_total = Some(0);
        assert!(!consistent.accounting_consistent());

        // Rows without reuse claims are vacuously consistent.
        consistent.bytes_new = None;
        assert!(consistent.accounting_consistent());
    }

    #[test]
    fn summaries_fold_min_median_max_over_one_arm_only() {
        let rows = vec![
            row("import", 0, 3.0),
            row("import", 1, 1.0),
            row("import", 2, 2.0),
            row("other", 0, 9.0),
        ];
        let summary = summarize("run", "import", &rows).expect("summary exists");
        assert_eq!(summary.reps, 3);
        assert_eq!(summary.wall_min_s, 1.0);
        assert_eq!(summary.wall_median_s, 2.0);
        assert_eq!(summary.wall_max_s, 3.0);
        assert!(summarize("run", "absent", &rows).is_none());
    }
}
