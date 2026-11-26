use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::code_validator::CodeValidator;
use crate::math_validator::MathValidator;
use crate::minhash_lsh::{MinHashLshConfig, MinHashLshEngine};
use crate::pedagogical_scorer::{CurriculumStage, PedagogicalScorer};
use crate::quality_filter::SystematicQualityFilter;
use crate::temporal_verifier::TemporalVerifier;

/// Global summary of the end-to-end pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSummary {
    pub total_input_documents: usize,
    pub deduplicated_documents: usize,
    pub duplicates_purged: usize,
    pub quality_clean_documents: usize,
    pub quality_defects_purged: usize,
    pub stage1_count: usize,
    pub stage2_count: usize,
    pub stage3_count: usize,
    pub stage4_count: usize,
    pub elapsed_seconds: f64,
    pub throughput_docs_per_sec: f64,
}

/// End-to-end multi-threaded data processing pipeline orchestrator.
pub struct ParallelPipeline {
    pub quality_filter: SystematicQualityFilter,
    pub dedup_engine: MinHashLshEngine,
    pub pedagogical_scorer: PedagogicalScorer,
    pub temporal_verifier: TemporalVerifier,
}

impl Default for ParallelPipeline {
    fn default() -> Self {
        Self {
            quality_filter: SystematicQualityFilter::default(),
            dedup_engine: MinHashLshEngine::new(MinHashLshConfig::default()),
            pedagogical_scorer: PedagogicalScorer::default(),
            temporal_verifier: TemporalVerifier::default(),
        }
    }
}

impl ParallelPipeline {
    /// Executes the full multi-threaded pipeline on a dataset file (CSV or JSONL).
    /// Writes clean documents partitioned by SmolLM2 curriculum stages.
    pub fn process_dataset(
        &self,
        input_path: &Path,
        output_dir: &Path,
        text_column: Option<&str>,
    ) -> Result<PipelineSummary, String> {
        let start_time = std::time::Instant::now();

        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed creating output directory {}: {e}", output_dir.display()))?;

        // 1. Read input documents into memory
        let file = File::open(input_path)
            .map_err(|e| format!("Failed opening {}: {e}", input_path.display()))?;
        let reader = BufReader::new(file);

        let is_jsonl = input_path.extension().and_then(|s| s.to_str()) == Some("jsonl");
        let mut documents: Vec<String> = Vec::new();
        let mut raw_rows: Vec<String> = Vec::new();

        if is_jsonl {
            for line_res in reader.lines() {
                let line = line_res.map_err(|e| format!("IO error: {e}"))?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    let col = text_column.unwrap_or("text");
                    if let Some(t) = val.get(col).and_then(|v| v.as_str()) {
                        documents.push(t.to_string());
                        raw_rows.push(line);
                    } else if let Some(t) = val.get("content").and_then(|v| v.as_str()) {
                        documents.push(t.to_string());
                        raw_rows.push(line);
                    }
                }
            }
        } else {
            // CSV
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .flexible(true)
                .from_reader(reader);

            let headers = rdr.headers().map_err(|e| format!("CSV header error: {e}"))?.clone();
            let col_idx = if let Some(target) = text_column {
                headers.iter().position(|h| h == target).ok_or_else(|| {
                    format!("Target column '{target}' not found in CSV headers: {:?}", headers)
                })?
            } else {
                headers
                    .iter()
                    .position(|h| {
                        let l = h.to_lowercase();
                        l == "text" || l == "content" || l == "body" || l == "document"
                    })
                    .unwrap_or(0)
            };

            for rec_res in rdr.records() {
                if let Ok(rec) = rec_res {
                    if let Some(field) = rec.get(col_idx) {
                        documents.push(field.to_string());
                        // Serialize back to CSV row representation
                        let mut wtr = csv::Writer::from_writer(vec![]);
                        let _ = wtr.write_record(&rec);
                        let s = String::from_utf8(wtr.into_inner().unwrap_or_default()).unwrap_or_default();
                        raw_rows.push(s);
                    }
                }
            }
        }

        let total_docs = documents.len();
        if total_docs == 0 {
            return Err("Dataset contains 0 valid documents".to_string());
        }

        // 2. MinHash LSH Global Deduplication
        let doc_slices: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let (dedup_mask, dedup_summary) = self.dedup_engine.deduplicate_texts(&doc_slices);

        // 3. Parallel Quality, Pedagogical & Verification Pipeline
        let stage1_docs = Mutex::new(Vec::new());
        let stage2_docs = Mutex::new(Vec::new());
        let stage3_docs = Mutex::new(Vec::new());
        let stage4_docs = Mutex::new(Vec::new());

        let kept_count = AtomicUsize::new(0);
        let quality_purged_count = AtomicUsize::new(0);

        (0..total_docs).into_par_iter().for_each(|idx| {
            if !dedup_mask[idx] {
                return; // Purged as near-duplicate
            }

            let text = &documents[idx];

            // A. Quality Filter
            let quality_eval = self.quality_filter.evaluate_document(text);
            if !quality_eval.is_clean {
                quality_purged_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            // B. Mathematical Formula Integrity
            let math_eval = MathValidator::validate_math_text(text);
            if !math_eval.is_valid {
                quality_purged_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            // C. Code AST Integrity (if contains code blocks)
            if text.contains("```") || text.starts_with("def ") || text.starts_with("fn ") {
                let code_eval = CodeValidator::validate_code(text);
                if !code_eval.is_valid {
                    quality_purged_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }

            // D. Temporal Interval Consistency
            let temporal_eval = self.temporal_verifier.verify_text(text);
            if !temporal_eval.is_temporally_consistent {
                quality_purged_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            // E. Pedagogical Readability & Curriculum Tier Routing
            let pedagogical = self.pedagogical_scorer.score_document(text);
            kept_count.fetch_add(1, Ordering::Relaxed);

            let row_payload = &raw_rows[idx];

            match pedagogical.recommended_stage {
                CurriculumStage::Stage1Fundamental => {
                    stage1_docs.lock().unwrap().push(row_payload.clone());
                }
                CurriculumStage::Stage2Intermediate => {
                    stage2_docs.lock().unwrap().push(row_payload.clone());
                }
                CurriculumStage::Stage3Advanced => {
                    stage3_docs.lock().unwrap().push(row_payload.clone());
                }
                CurriculumStage::Stage4Research => {
                    stage4_docs.lock().unwrap().push(row_payload.clone());
                }
            }
        });

        // 4. Flush Clean Documents to Stage Partitions
        let s1 = stage1_docs.into_inner().unwrap();
        let s2 = stage2_docs.into_inner().unwrap();
        let s3 = stage3_docs.into_inner().unwrap();
        let s4 = stage4_docs.into_inner().unwrap();

        let ext = if is_jsonl { "jsonl" } else { "csv" };
        let write_stage = |stage_name: &str, rows: &[String]| -> Result<(), String> {
            if rows.is_empty() {
                return Ok(());
            }
            let path = output_dir.join(format!("{}.{}", stage_name, ext));
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("Failed opening {}: {e}", path.display()))?;
            for row in rows {
                f.write_all(row.as_bytes()).map_err(|e| format!("Write error: {e}"))?;
                if !row.ends_with('\n') {
                    f.write_all(b"\n").map_err(|e| format!("Write error: {e}"))?;
                }
            }
            Ok(())
        };

        write_stage("stage1_fundamental", &s1)?;
        write_stage("stage2_intermediate", &s2)?;
        write_stage("stage3_advanced", &s3)?;
        write_stage("stage4_research", &s4)?;

        let elapsed = start_time.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            total_docs as f64 / elapsed
        } else {
            0.0
        };

        let final_kept = kept_count.load(Ordering::Relaxed);
        let final_purged = quality_purged_count.load(Ordering::Relaxed);

        Ok(PipelineSummary {
            total_input_documents: total_docs,
            deduplicated_documents: dedup_summary.unique_documents_kept,
            duplicates_purged: dedup_summary.duplicate_documents_purged,
            quality_clean_documents: final_kept,
            quality_defects_purged: final_purged,
            stage1_count: s1.len(),
            stage2_count: s2.len(),
            stage3_count: s3.len(),
            stage4_count: s4.len(),
            elapsed_seconds: elapsed,
            throughput_docs_per_sec: throughput,
        })
    }
}
