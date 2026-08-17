//! The state assertion behind the hard cut A→B: **whole-tree
//! materialization does not exist in any API.**
//!
//! The projection replaces it outright — a snapshot tree is directories,
//! symlinks and stubs, and nothing in this repository offers to write a
//! snapshot's bytes into a directory. The single-file escape hatch
//! (`extract`) survives and is deliberately NOT fenced: it is one file, on
//! purpose, and §9 of the layout doc prices it.
//!
//! This is a source fence rather than a behavioural test because the claim is
//! about absence. It fails the moment anyone defines a whole-tree
//! materializer in shipped code, in any of the three languages.

use std::path::{Path, PathBuf};

/// Definition keywords in the three languages this repository ships.
const DEFINITIONS: [&str; 3] = ["fn ", "def ", "func "];

/// Names that would mean "write a whole snapshot's bytes into a directory".
/// `extract` alone is absent on purpose: that is the priced one-file hatch.
const FORBIDDEN: [&str; 6] = [
    "materialize",
    "materialise",
    "extract_tree",
    "extract_all",
    "extracttree",
    "extractall",
];

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
    for path in &sources {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            let lowered = line.to_lowercase();
            let trimmed = lowered.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
                continue;
            }
            let Some(definition) = DEFINITIONS
                .iter()
                .filter_map(|keyword| lowered.find(keyword).map(|at| at + keyword.len()))
                .min()
            else {
                continue;
            };
            let name: String = lowered[definition..]
                .chars()
                .take_while(|character| character.is_alphanumeric() || *character == '_')
                .collect();
            if FORBIDDEN.iter().any(|banned| name.contains(banned)) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(&root).unwrap_or(path).display(),
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
}
