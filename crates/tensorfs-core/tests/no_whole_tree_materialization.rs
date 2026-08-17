//! The state assertion behind the hard cut A→B: **whole-tree
//! materialization does not exist in any API.**
//!
//! The projection replaces it outright — a snapshot tree is directories,
//! symlinks and stubs, and nothing in this repository offers to write a
//! snapshot's bytes into a directory. The single-file escape hatch survives
//! on purpose: it is tier 3 of the access ladder (§9), it writes ONE file,
//! and it is now spelled `materialize()` — Paul's 2026-08-17 #1303 ruling
//! renamed it from `extract()`.
//!
//! That rename is why this fence is name-shaped rather than a blunt ban on
//! the word: the hatch now OWNS the word "materialize", so the fence pins it
//! instead of forbidding it. Two rules, and both are red-proved below:
//!
//! 1. a definition naming materialization is an offender unless it is
//!    EXACTLY the sanctioned one-file hatch, at its one sanctioned path —
//!    so `materialize_repository`, `materialize_tree`, `materialise_all` and
//!    a second copy of the hatch in another module are all refused;
//! 2. `extract*` is the RETIRED spelling and may not come back, in any
//!    language, under any suffix.
//!
//! This is a source fence rather than a behavioural test because the claim is
//! about absence. It fails the moment anyone defines a whole-tree
//! materializer in shipped code, in any of the three languages.

use std::path::{Path, PathBuf};

/// Definition keywords in the three languages this repository ships.
const DEFINITIONS: [&str; 3] = ["fn ", "def ", "func "];

/// The fragment every spelling of "materialize" shares — including
/// "materialise", and every suffixed form. A definition containing it is an
/// offender unless it is [`HATCH_NAME`] at [`HATCH_PATH`].
const MATERIALIZES: &str = "materiali";

/// Names that would mean "write a whole snapshot's bytes into a directory"
/// under the retired spelling, plus the retired spelling itself. `extract`
/// is a prefix match: `extract`, `extract_tree` and `extract_all` all go.
const RETIRED_SPELLING: &str = "extract";

/// The ONE sanctioned materializer: one file, atomic, verified, tier 3.
const HATCH_NAME: &str = "materialize";

/// …and the one module allowed to define it. A second definition elsewhere
/// is a second hatch, which is how a bounded escape hatch becomes a data
/// plane again.
const HATCH_PATH: &str = "python/src/tensorfs/tensors.py";

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels under the repository root")
        .to_path_buf()
}

/// Only SHIPPED surface. Tests, benchmarks and docs may name the deleted API
/// freely — `python/benchmarks/direct_vs_materialize.py` reproduces it on
/// purpose to measure what the projection replaced.
fn shipped_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.join("crates"), root.join("python").join("src")];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // `crates/<name>/src` only: a crate's tests/ and examples/
                // are not API.
                if path.file_name().is_some_and(|name| {
                    name == "tests" || name == "examples" || name == "benchmarks"
                }) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path
                .extension()
                .is_some_and(|ext| ext == "rs" || ext == "py")
            {
                found.push(path);
            }
        }
    }
    // The Go decoder is a flat package at the repository root.
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if path.extension().is_some_and(|ext| ext == "go") && !name.ends_with("_test.go") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// The name a definition line declares, if it declares one.
fn defined_name(line: &str) -> Option<String> {
    let lowered = line.to_lowercase();
    let trimmed = lowered.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
        return None;
    }
    let definition = DEFINITIONS
        .iter()
        .filter_map(|keyword| lowered.find(keyword).map(|at| at + keyword.len()))
        .min()?;
    let name: String = lowered[definition..]
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Why this definition is refused, or `None` when it is allowed.
///
/// `relative` is repository-relative and POSIX-separated, because the one
/// exemption is a PATH: the hatch may exist exactly once, where the ladder's
/// documentation says it lives.
fn verdict(relative: &str, line: &str) -> Option<String> {
    let name = defined_name(line)?;
    if name.starts_with(RETIRED_SPELLING) {
        return Some(format!(
            "`{name}` uses the RETIRED spelling: the one-file hatch was renamed \
             `extract()` -> `materialize()` (Paul's #1303 ruling, 2026-08-17)"
        ));
    }
    if name.contains(MATERIALIZES) {
        if name == HATCH_NAME && relative == HATCH_PATH {
            return None;
        }
        return Some(format!(
            "`{name}` is a materializer that is not the one sanctioned hatch \
             (`{HATCH_NAME}` at `{HATCH_PATH}`)"
        ));
    }
    None
}

#[test]
fn the_fence_refuses_a_tree_materializer_and_the_retired_spelling() {
    // Red proof, in-tree: every rule fires on a planted line, and the
    // sanctioned hatch does not. A fence whose rules are only ever exercised
    // by a green tree cannot say whether it still binds.
    let planted: [(&str, &str, bool); 9] = [
        (HATCH_PATH, "    def materialize(", false),
        // The same name, one module over, is a second hatch.
        ("python/src/tensorfs/local.py", "    def materialize(", true),
        (HATCH_PATH, "    def materialize_repository(", true),
        (HATCH_PATH, "    def materialize_tree(", true),
        (
            "crates/tensorfs-core/src/lib.rs",
            "pub fn materialise_all(",
            true,
        ),
        ("store.go", "func MaterializeSnapshot(", true),
        // The retired spelling, bare and suffixed.
        (HATCH_PATH, "    def extract(", true),
        (
            "crates/tensorfs-core/src/lib.rs",
            "pub fn extract_tree(",
            true,
        ),
        // An ordinary definition that merely mentions the words is not one.
        (
            HATCH_PATH,
            "    def read_range(self, path: str) -> bytes:",
            false,
        ),
    ];
    for (path, line, expected) in planted {
        assert_eq!(
            verdict(path, line).is_some(),
            expected,
            "the fence's verdict on {line:?} at {path} is wrong"
        );
    }
}

#[test]
fn no_shipped_api_defines_a_whole_tree_materializer() {
    let root = repository_root();
    let sources = shipped_sources(&root);
    assert!(
        sources.len() > 10,
        "the fence scanned {} files under {}; it cannot vouch for a surface it never read",
        sources.len(),
        root.display()
    );
    assert!(
        sources
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "rs")),
        "the fence must reach the Rust surface"
    );
    assert!(
        sources
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "py")),
        "the fence must reach the Python surface"
    );
    assert!(
        sources
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "go")),
        "the fence must reach the Go surface"
    );

    let mut offenders = Vec::new();
    let mut hatch_definitions = 0usize;
    for path in &sources {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        for (number, line) in text.lines().enumerate() {
            if relative == HATCH_PATH && defined_name(line).as_deref() == Some(HATCH_NAME) {
                hatch_definitions += 1;
            }
            if let Some(reason) = verdict(&relative, line) {
                offenders.push(format!(
                    "{relative}:{}: {reason}: {}",
                    number + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "whole-tree materialization is back in the shipped surface:\n  {}\n\
         The projection replaced it (docs/mixed-cas-layout.md §9). If this is a \
         deliberate reversal, that is a design decision, not a test to relax.",
        offenders.join("\n  ")
    );

    // The exemption must describe something. If the hatch moves or is renamed
    // again, the path rule above silently stops exempting anything and this
    // fence would pass while vouching for nothing.
    assert_eq!(
        hatch_definitions, 1,
        "the ladder's tier-3 hatch is `{HATCH_NAME}` at `{HATCH_PATH}`, defined \
         exactly once; found {hatch_definitions} definitions. Moving it means \
         updating this fence's exemption in the same change."
    );
}
