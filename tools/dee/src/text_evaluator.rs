use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::profiler::{median, numeric_summary, NumericSummary};

/// Common high-frequency English stopwords for natural language density analysis.
const TOP_ENGLISH_STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "even", "first", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "like", "make", "many", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "us", "use", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
];

/// Non-neural deterministic text metrics computed directly from raw text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextMetrics {
    pub char_count: usize,
    pub word_count: usize,
    pub line_count: usize,
    pub avg_word_length: f64,
    /// Shannon entropy in bits per character. Natural English is typically 3.8 - 4.6.
    pub shannon_entropy: f64,
    /// Ratio of recognized natural English stopwords to total words. Natural prose is ~0.30 - 0.45.
    pub stopword_ratio: f64,
    /// Ratio of duplicate non-empty lines to total non-empty lines. Detects template spam.
    pub line_repetition_ratio: f64,
    /// Ratio of lines that end with standard sentence punctuation (. ! ? " ').
    pub terminal_punctuation_ratio: f64,
    /// Ratio of whitespace characters to total characters.
    pub whitespace_ratio: f64,
    /// Ratio of digit characters (0-9) to total characters.
    pub digit_ratio: f64,
    /// Ratio of uppercase characters to total alphabetic characters.
    pub uppercase_ratio: f64,
    /// Ratio of non-ASCII characters to total characters.
    pub non_ascii_ratio: f64,
    /// Count of HTML tags or unstripped markup artifacts.
    pub html_artifact_count: usize,
    /// Count of Unicode replacement characters (\u{fffd}).
    pub unicode_replacement_count: usize,
}

impl TextMetrics {
    pub fn compute(text: &str) -> Self {
        let char_count = text.chars().count();
        if char_count == 0 {
            return Self {
                char_count: 0,
                word_count: 0,
                line_count: 0,
                avg_word_length: 0.0,
                shannon_entropy: 0.0,
                stopword_ratio: 0.0,
                line_repetition_ratio: 0.0,
                terminal_punctuation_ratio: 0.0,
                whitespace_ratio: 0.0,
                digit_ratio: 0.0,
                uppercase_ratio: 0.0,
                non_ascii_ratio: 0.0,
                html_artifact_count: 0,
                unicode_replacement_count: 0,
            };
        }

        let mut char_freq: HashMap<char, usize> = HashMap::new();
        let mut whitespace_count = 0usize;
        let mut digit_count = 0usize;
        let mut alpha_count = 0usize;
        let mut uppercase_count = 0usize;
        let mut non_ascii_count = 0usize;
        let mut unicode_replacement_count = 0usize;

        for c in text.chars() {
            *char_freq.entry(c).or_insert(0) += 1;
            if c.is_whitespace() {
                whitespace_count += 1;
            }
            if c.is_ascii_digit() {
                digit_count += 1;
            }
            if c.is_alphabetic() {
                alpha_count += 1;
                if c.is_uppercase() {
                    uppercase_count += 1;
                }
            }
            if !c.is_ascii() {
                non_ascii_count += 1;
            }
            if c == '\u{fffd}' {
                unicode_replacement_count += 1;
            }
        }

        // Shannon entropy: H = -sum(p * log2(p))
        let total_chars_f = char_count as f64;
        let mut shannon_entropy = 0.0f64;
        for &cnt in char_freq.values() {
            let p = cnt as f64 / total_chars_f;
            if p > 0.0 {
                shannon_entropy -= p * p.log2();
            }
        }

        // Words & Stopwords
        let words: Vec<&str> = text
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|w| !w.is_empty())
            .collect();
        let word_count = words.len();

        let stopword_set: HashSet<&str> = TOP_ENGLISH_STOPWORDS.iter().copied().collect();
        let mut stopword_count = 0usize;
        let mut total_word_len = 0usize;

        for &w in &words {
            total_word_len += w.chars().count();
            let lower = w.to_ascii_lowercase();
            if stopword_set.contains(lower.as_str()) {
                stopword_count += 1;
            }
        }

        let avg_word_length = if word_count > 0 {
            total_word_len as f64 / word_count as f64
        } else {
            0.0
        };

        let stopword_ratio = if word_count > 0 {
            stopword_count as f64 / word_count as f64
        } else {
            0.0
        };

        // Lines and Repetitions
        let raw_lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        let line_count = raw_lines.len();

        let mut line_seen = HashSet::new();
        let mut duplicate_lines = 0usize;
        let mut lines_with_terminal_punct = 0usize;

        for &l in &raw_lines {
            if !line_seen.insert(l) {
                duplicate_lines += 1;
            }
            if let Some(last_ch) = l.chars().last() {
                if matches!(last_ch, '.' | '!' | '?' | '"' | '\'' | ':' | ';') {
                    lines_with_terminal_punct += 1;
                }
            }
        }

        let line_repetition_ratio = if line_count > 0 {
            duplicate_lines as f64 / line_count as f64
        } else {
            0.0
        };

        let terminal_punctuation_ratio = if line_count > 0 {
            lines_with_terminal_punct as f64 / line_count as f64
        } else {
            0.0
        };

        // HTML & Markup artifacts
        let lower_text = text.to_lowercase();
        let html_patterns = [
            "<script", "</script>", "<style", "<div", "</div>", "href=", "class=",
            "&nbsp;", "&amp;", "&quot;", "&lt;", "&gt;", "<span", "</span>", "<!--",
        ];
        let mut html_artifact_count = 0usize;
        for pat in html_patterns {
            html_artifact_count += lower_text.matches(pat).count();
        }

        Self {
            char_count,
            word_count,
            line_count,
            avg_word_length,
            shannon_entropy,
            stopword_ratio,
            line_repetition_ratio,
            terminal_punctuation_ratio,
            whitespace_ratio: whitespace_count as f64 / total_chars_f,
            digit_ratio: digit_count as f64 / total_chars_f,
            uppercase_ratio: if alpha_count > 0 {
                uppercase_count as f64 / alpha_count as f64
            } else {
                0.0
            },
            non_ascii_ratio: non_ascii_count as f64 / total_chars_f,
            html_artifact_count,
            unicode_replacement_count,
        }
    }
}

/// A specific text quality anomaly flagged during independent evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAnomaly {
    pub row_index: usize,
    pub metric: String,
    pub observed_value: f64,
    pub robust_z: f64,
    pub reason: String,
    pub text_preview: String,
}

/// Dataset-wide aggregate summary of raw text quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextQualityReport {
    pub source: String,
    pub text_column: String,
    pub total_documents: usize,
    pub char_count: NumericSummary,
    pub word_count: NumericSummary,
    pub shannon_entropy: NumericSummary,
    pub stopword_ratio: NumericSummary,
    pub line_repetition_ratio: NumericSummary,
    pub terminal_punctuation_ratio: NumericSummary,
    pub non_ascii_ratio: NumericSummary,
    pub html_artifact_total: usize,
    pub unicode_replacement_total: usize,
    pub anomalies: Vec<TextAnomaly>,
}

impl TextQualityReport {
    pub fn to_pretty_text(&self) -> String {
        let mut out = String::new();
        out.push_str("================================================================================\n");
        out.push_str("DEE INDEPENDENT TEXT QUALITY EVALUATION REPORT\n");
        out.push_str(&format!("Source: {} | Column: {}\n", self.source, self.text_column));
        out.push_str(&format!("Documents Evaluated: {}\n", self.total_documents));
        out.push_str("================================================================================\n\n");

        out.push_str("TEXT METRIC DISTRIBUTIONS:\n");
        out.push_str(&format!(
            "  * Word Count       : min={:?} max={:?} median={:?} MAD={:?}\n",
            self.word_count.min, self.word_count.max, self.word_count.median, self.word_count.mad
        ));
        out.push_str(&format!(
            "  * Shannon Entropy  : min={:?} max={:?} median={:?} MAD={:?} (normal: 3.8 - 4.6)\n",
            self.shannon_entropy.min, self.shannon_entropy.max, self.shannon_entropy.median, self.shannon_entropy.mad
        ));
        out.push_str(&format!(
            "  * Stopword Density : min={:?} max={:?} median={:?} MAD={:?} (normal prose: 0.30 - 0.45)\n",
            self.stopword_ratio.min, self.stopword_ratio.max, self.stopword_ratio.median, self.stopword_ratio.mad
        ));
        out.push_str(&format!(
            "  * Line Repetition  : min={:?} max={:?} median={:?} MAD={:?} (high = boilerplate)\n",
            self.line_repetition_ratio.min, self.line_repetition_ratio.max, self.line_repetition_ratio.median, self.line_repetition_ratio.mad
        ));
        out.push_str(&format!(
            "  * Sentence Ending  : min={:?} max={:?} median={:?} MAD={:?}\n",
            self.terminal_punctuation_ratio.min, self.terminal_punctuation_ratio.max, self.terminal_punctuation_ratio.median, self.terminal_punctuation_ratio.mad
        ));
        out.push_str(&format!(
            "  * Non-ASCII Ratio  : min={:?} max={:?} median={:?} MAD={:?}\n",
            self.non_ascii_ratio.min, self.non_ascii_ratio.max, self.non_ascii_ratio.median, self.non_ascii_ratio.mad
        ));
        out.push_str(&format!(
            "  * Global Artifacts : HTML Artifacts={}, Unicode Replacement Chars (\\ufffd)={}\n\n",
            self.html_artifact_total, self.unicode_replacement_total
        ));

        out.push_str(&format!(
            "ANOMALIES FLAGGED BY DEE (Total: {}):\n",
            self.anomalies.len()
        ));
        if self.anomalies.is_empty() {
            out.push_str("  None. All documents fall within expected statistical quality boundaries.\n");
        } else {
            for (idx, a) in self.anomalies.iter().take(20).enumerate() {
                out.push_str(&format!(
                    "  {:03}. [Row {}] {}={:.4} (robust-z={:.2})\n       Reason: {}\n       Preview: \"{}\"\n",
                    idx + 1,
                    a.row_index,
                    a.metric,
                    a.observed_value,
                    a.robust_z,
                    a.reason,
                    a.text_preview
                ));
            }
            if self.anomalies.len() > 20 {
                out.push_str(&format!(
                    "  ... and {} additional anomalies recorded.\n",
                    self.anomalies.len() - 20
                ));
            }
        }
        out
    }

    pub fn to_json_file(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(file, self).map_err(|e| e.to_string())
    }
}

pub fn extract_text_from_json_value(val: &serde_json::Value, key: &str) -> Option<String> {
    if let Some(v) = val.get(key) {
        if let Some(s) = v.as_str() {
            return Some(s.to_string());
        }
        if key == "messages" && v.is_array() {
            let mut parts = Vec::new();
            if let Some(arr) = v.as_array() {
                for item in arr {
                    if let Some(c) = item.get("content").and_then(|c| c.as_str()) {
                        parts.push(c);
                    }
                }
            }
            if !parts.is_empty() {
                return Some(parts.join("\n"));
            }
        }
    }
    None
}

/// Independently evaluates raw text content in a CSV or JSONL dataset without relying on external scores.
pub fn evaluate_text_column(
    dataset_path: &Path,
    text_column_name: Option<&str>,
    threshold: f64,
) -> Result<TextQualityReport, String> {
    let is_jsonl = dataset_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false);

    let mut metrics_list = Vec::new();
    let mut previews = Vec::new();
    let col_name: String;

    if is_jsonl {
        let file = File::open(dataset_path).map_err(|e| format!("JSONL file read error: {e}"))?;
        let reader = BufReader::new(file);
        let mut determined_col: Option<String> = text_column_name.map(|s| s.to_string());

        for (idx, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("JSONL line error at {idx}: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("JSON parse error at line {}: {e}", idx + 1))?;

            let text_val = if let Some(ref col) = determined_col {
                extract_text_from_json_value(&val, col)
            } else {
                let candidates = [
                    "text", "content", "response", "output", "prompt", "instruction",
                    "document", "body", "messages",
                ];
                let mut found_text = None;
                for c in candidates {
                    if let Some(t) = extract_text_from_json_value(&val, c) {
                        determined_col = Some(c.to_string());
                        found_text = Some(t);
                        break;
                    }
                }
                found_text
            }
            .unwrap_or_default();

            let m = TextMetrics::compute(&text_val);
            let preview: String = text_val.chars().take(80).collect::<String>().replace('\n', " ");
            metrics_list.push(m);
            previews.push(preview);
        }

        col_name = determined_col.unwrap_or_else(|| "text".to_string());
    } else {
        let mut rdr = csv::Reader::from_path(dataset_path)
            .map_err(|e| format!("CSV read error in text evaluation: {e}"))?;
        let headers = rdr
            .headers()
            .map_err(|e| format!("CSV header error: {e}"))?
            .clone();

        // Auto-detect text column if not provided
        let text_col_idx = if let Some(name) = text_column_name {
            headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or_else(|| format!("Column '{name}' not found in CSV"))?
        } else {
            // Search for common text column names
            let candidates = ["text", "content", "document", "body", "raw_text", "article"];
            let mut found = None;
            for c in candidates {
                if let Some(idx) = headers.iter().position(|h| h.eq_ignore_ascii_case(c)) {
                    found = Some(idx);
                    break;
                }
            }
            found.ok_or_else(|| {
                "No text column specified and could not auto-detect ('text', 'content', etc.)".to_string()
            })?
        };

        col_name = headers.get(text_col_idx).unwrap().to_string();

        for record in rdr.records() {
            let record = record.map_err(|e| format!("CSV row error: {e}"))?;
            let text = record.get(text_col_idx).unwrap_or("");
            let m = TextMetrics::compute(text);
            let preview: String = text.chars().take(80).collect::<String>().replace('\n', " ");
            metrics_list.push(m);
            previews.push(preview);
        }
    }

    let n = metrics_list.len();
    if n == 0 {
        return Err("Dataset contains 0 rows".to_string());
    }

    // Extract feature vectors
    let char_counts: Vec<f64> = metrics_list.iter().map(|m| m.char_count as f64).collect();
    let word_counts: Vec<f64> = metrics_list.iter().map(|m| m.word_count as f64).collect();
    let entropies: Vec<f64> = metrics_list.iter().map(|m| m.shannon_entropy).collect();
    let stopwords: Vec<f64> = metrics_list.iter().map(|m| m.stopword_ratio).collect();
    let line_reps: Vec<f64> = metrics_list.iter().map(|m| m.line_repetition_ratio).collect();
    let term_puncts: Vec<f64> = metrics_list.iter().map(|m| m.terminal_punctuation_ratio).collect();
    let non_asciis: Vec<f64> = metrics_list.iter().map(|m| m.non_ascii_ratio).collect();

    let mut html_artifact_total = 0usize;
    let mut unicode_replacement_total = 0usize;
    for m in &metrics_list {
        html_artifact_total += m.html_artifact_count;
        unicode_replacement_total += m.unicode_replacement_count;
    }

    // Compute MAD outlier detectors for key dimensions
    let mut anomalies = Vec::new();

    let check_outliers = |values: &[f64],
                          metric_name: &str,
                          check_low: bool,
                          check_high: bool,
                          low_reason: &str,
                          high_reason: &str,
                          anomalies_out: &mut Vec<TextAnomaly>| {
        if let Some(med) = median(values) {
            let devs: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
            let mad = median(&devs).unwrap_or(0.0);
            if mad > f64::EPSILON {
                for (i, &val) in values.iter().enumerate() {
                    let rz = 0.6745 * (val - med) / mad;
                    if check_high && rz >= threshold {
                        anomalies_out.push(TextAnomaly {
                            row_index: i + 2,
                            metric: metric_name.to_string(),
                            observed_value: val,
                            robust_z: rz,
                            reason: high_reason.to_string(),
                            text_preview: previews[i].clone(),
                        });
                    } else if check_low && rz <= -threshold {
                        anomalies_out.push(TextAnomaly {
                            row_index: i + 2,
                            metric: metric_name.to_string(),
                            observed_value: val,
                            robust_z: rz,
                            reason: low_reason.to_string(),
                            text_preview: previews[i].clone(),
                        });
                    }
                }
            }
        }
    };

    // 1. Shannon entropy: unusually high (encrypted/gibberish) or unusually low (repetitive loops)
    check_outliers(
        &entropies,
        "shannon_entropy",
        true,
        true,
        "Abnormally low character entropy: repetitive token loop or whitespace padding",
        "Abnormally high character entropy: possible random hex, base64, or encrypted dump",
        &mut anomalies,
    );

    // 2. Stopword density: unusually low indicates SEO keyword spam, scraped lists, or code
    check_outliers(
        &stopwords,
        "stopword_ratio",
        true,
        false,
        "Abnormally low natural English stopword density: likely catalog list, code dump, or keyword spam",
        "",
        &mut anomalies,
    );

    // 3. Line repetition: high indicates repeated boilerplate / templates
    check_outliers(
        &line_reps,
        "line_repetition_ratio",
        false,
        true,
        "",
        "High line-level duplication: repeated website navigation headers, footers, or boilerplate",
        &mut anomalies,
    );

    // 4. Non-ASCII ratio: abnormally high indicates encoding mismatch or mixed non-English text
    check_outliers(
        &non_asciis,
        "non_ascii_ratio",
        false,
        true,
        "",
        "High non-ASCII character fraction: possible encoding corruption or non-English script",
        &mut anomalies,
    );

    // 5. Flag Unicode replacement characters (\ufffd) directly as severe encoding corruptions
    for (i, m) in metrics_list.iter().enumerate() {
        if m.unicode_replacement_count > 0 {
            anomalies.push(TextAnomaly {
                row_index: i + 2,
                metric: "unicode_replacement_count".to_string(),
                observed_value: m.unicode_replacement_count as f64,
                robust_z: 99.0,
                reason: format!("Found {} Unicode replacement characters (\\ufffd) indicating severe encoding corruption", m.unicode_replacement_count),
                text_preview: previews[i].clone(),
            });
        }
        if m.html_artifact_count >= 10 {
            anomalies.push(TextAnomaly {
                row_index: i + 2,
                metric: "html_artifact_count".to_string(),
                observed_value: m.html_artifact_count as f64,
                robust_z: 10.0,
                reason: format!("High density of raw HTML markup artifacts ({}) detected in text", m.html_artifact_count),
                text_preview: previews[i].clone(),
            });
        }
    }

    Ok(TextQualityReport {
        source: dataset_path.display().to_string(),
        text_column: col_name,
        total_documents: n,
        char_count: numeric_summary(&char_counts, 0),
        word_count: numeric_summary(&word_counts, 0),
        shannon_entropy: numeric_summary(&entropies, 0),
        stopword_ratio: numeric_summary(&stopwords, 0),
        line_repetition_ratio: numeric_summary(&line_reps, 0),
        terminal_punctuation_ratio: numeric_summary(&term_puncts, 0),
        non_ascii_ratio: numeric_summary(&non_asciis, 0),
        html_artifact_total,
        unicode_replacement_total,
        anomalies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_metrics_natural_prose() {
        let prose = "This is a natural English sentence explaining the core concepts of biology. It contains standard words, punctuation, and structure.";
        let m = TextMetrics::compute(prose);
        assert!(m.word_count > 10);
        assert!(m.stopword_ratio > 0.25);
        assert!(m.shannon_entropy > 3.5 && m.shannon_entropy < 5.0);
        assert_eq!(m.unicode_replacement_count, 0);
        assert_eq!(m.html_artifact_count, 0);
    }

    #[test]
    fn test_text_metrics_detects_low_entropy_and_repetition() {
        let loop_text = "repeat repeat repeat repeat\nrepeat repeat repeat repeat\nrepeat repeat repeat repeat";
        let m = TextMetrics::compute(loop_text);
        assert!(m.line_repetition_ratio > 0.5);
    }

    #[test]
    fn test_text_metrics_detects_html_and_replacement_chars() {
        let bad_text = "Hello <script>alert(1)</script> world \u{fffd} with <div>content</div>";
        let m = TextMetrics::compute(bad_text);
        assert!(m.html_artifact_count >= 2);
        assert_eq!(m.unicode_replacement_count, 1);
    }
}
