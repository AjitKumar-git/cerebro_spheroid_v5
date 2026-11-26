use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs::File, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericSummary {
    pub count: usize,
    pub missing: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub mad: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub non_empty: usize,
    pub missing: usize,
    pub inferred_kind: String,
    pub numeric: Option<NumericSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetReport {
    pub source: String,
    pub rows: usize,
    pub columns: usize,
    pub exact_duplicate_rows: usize,
    pub profiles: Vec<ColumnProfile>,
    pub principles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RarityCandidate {
    pub row_index: usize,
    pub column: String,
    pub value: f64,
    pub robust_z: f64,
    pub interpretation: String,
}

impl DatasetReport {
    pub fn to_pretty_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "DEE Dataset Inspection\nSource: {}\nRows: {}\nColumns: {}\nExact duplicate rows: {}\n\n",
            self.source, self.rows, self.columns, self.exact_duplicate_rows
        ));
        out.push_str("Column profiles:\n");
        for p in &self.profiles {
            out.push_str(&format!(
                "  - {} | kind={} | non-empty={} | missing={}\n",
                p.name, p.inferred_kind, p.non_empty, p.missing
            ));
            if let Some(n) = &p.numeric {
                out.push_str(&format!(
                    "      numeric: count={} min={:?} max={:?} mean={:?} median={:?} MAD={:?}\n",
                    n.count, n.min, n.max, n.mean, n.median, n.mad
                ));
            }
        }
        out.push_str("\nInterpretation safeguards:\n");
        for p in &self.principles {
            out.push_str(&format!("  * {p}\n"));
        }
        out
    }

    pub fn to_json_file(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(file, self).map_err(|e| e.to_string())
    }
}

pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        Some((v[mid - 1] + v[mid]) / 2.0)
    } else {
        Some(v[mid])
    }
}

pub fn numeric_summary(values: &[f64], missing: usize) -> NumericSummary {
    if values.is_empty() {
        return NumericSummary {
            count: 0,
            missing,
            min: None,
            max: None,
            mean: None,
            median: None,
            mad: None,
        };
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let med = median(values);
    let mad = med.map(|m| {
        median(&values.iter().map(|x| (x - m).abs()).collect::<Vec<_>>()).unwrap_or(0.0)
    });
    NumericSummary {
        count: values.len(),
        missing,
        min: Some(min),
        max: Some(max),
        mean: Some(mean),
        median: med,
        mad,
    }
}

pub fn inspect_csv(path: &Path) -> Result<DatasetReport, String> {
    let mut rdr = csv::Reader::from_path(path).map_err(|e| format!("CSV read error: {e}"))?;
    let headers = rdr.headers().map_err(|e| format!("CSV header error: {e}"))?.clone();
    let mut columns: Vec<Vec<String>> = vec![Vec::new(); headers.len()];
    let mut rows = 0usize;
    let mut row_fingerprints: HashSet<Vec<String>> = HashSet::new();
    let mut duplicates = 0usize;

    for record in rdr.records() {
        let record = record.map_err(|e| format!("CSV row error: {e}"))?;
        rows += 1;
        let row: Vec<String> = (0..headers.len())
            .map(|i| record.get(i).unwrap_or("").trim().to_string())
            .collect();
        if !row_fingerprints.insert(row.clone()) {
            duplicates += 1;
        }
        for (i, value) in row.into_iter().enumerate() {
            columns[i].push(value);
        }
    }

    let profiles = headers
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let vals = &columns[i];
            let missing = vals.iter().filter(|v| v.is_empty()).count();
            let non_empty_vals: Vec<&String> = vals.iter().filter(|v| !v.is_empty()).collect();
            let parsed: Vec<f64> = non_empty_vals
                .iter()
                .filter_map(|v| v.parse::<f64>().ok())
                .filter(|x| x.is_finite())
                .collect();
            let is_numeric = !non_empty_vals.is_empty() && parsed.len() == non_empty_vals.len();
            let kind = if non_empty_vals.is_empty() {
                "empty"
            } else if is_numeric {
                "numeric"
            } else if non_empty_vals.iter().all(|v| v.parse::<bool>().is_ok()) {
                "boolean-like"
            } else {
                "categorical/text"
            };
            ColumnProfile {
                name: name.to_string(),
                non_empty: vals.len() - missing,
                missing,
                inferred_kind: kind.to_string(),
                numeric: if is_numeric {
                    Some(numeric_summary(&parsed, missing))
                } else {
                    None
                },
            }
        })
        .collect();

    Ok(DatasetReport {
        source: path.display().to_string(),
        rows,
        columns: headers.len(),
        exact_duplicate_rows: duplicates,
        profiles,
        principles: vec![
            "Marginal rarity is not a corruption verdict.".to_string(),
            "Original input is read-only; this tool never edits the source.".to_string(),
            "Profile statistics do not establish causal or conditional relationships.".to_string(),
            "Insufficient evidence should remain unresolved.".to_string(),
        ],
    })
}

pub fn investigate_csv(path: &Path, threshold: f64) -> Result<Vec<RarityCandidate>, String> {
    let mut rdr = csv::Reader::from_path(path).map_err(|e| format!("CSV read error: {e}"))?;
    let headers = rdr.headers().map_err(|e| format!("CSV header error: {e}"))?.clone();
    let mut columns: Vec<Vec<Option<f64>>> = vec![Vec::new(); headers.len()];
    for record in rdr.records() {
        let record = record.map_err(|e| format!("CSV row error: {e}"))?;
        for i in 0..headers.len() {
            let s = record.get(i).unwrap_or("").trim();
            columns[i].push(if s.is_empty() {
                None
            } else {
                s.parse::<f64>().ok().filter(|x| x.is_finite())
            });
        }
    }

    let mut candidates = Vec::new();
    for (col_idx, values) in columns.iter().enumerate() {
        let present: Vec<f64> = values.iter().flatten().copied().collect();
        if present.len() < 5 {
            continue;
        }
        let med = match median(&present) {
            Some(v) => v,
            None => continue,
        };
        let deviations: Vec<f64> = present.iter().map(|v| (v - med).abs()).collect();
        let mad = median(&deviations).unwrap_or(0.0);
        if mad <= f64::EPSILON {
            continue;
        }
        for (row_idx, value) in values.iter().enumerate() {
            if let Some(x) = value {
                let rz = 0.6745 * (x - med) / mad;
                if rz.abs() >= threshold {
                    candidates.push(RarityCandidate {
                        row_index: row_idx + 2, // CSV header is line 1; first data line is 2.
                        column: headers.get(col_idx).unwrap_or("").to_string(),
                        value: *x,
                        robust_z: rz,
                        interpretation:
                            "Marginally unusual; investigate context and dependencies. Not a corruption verdict."
                                .to_string(),
                    });
                }
            }
        }
    }
    Ok(candidates)
}
