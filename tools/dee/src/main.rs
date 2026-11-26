use clap::{Parser, Subcommand};
use dee::{
    evaluate_text_column, filter_dataset, generate_evidence_ledger, infer_schema_from_csv,
    inspect_csv, investigate_csv, validate_csv, CodeValidator, GeopoliticalBalancer,
    MathValidator, MinHashLshConfig, MinHashLshEngine, ParallelPipeline,
    PedagogicalScorer, PseudoscienceFilter, SchemaContract, SystematicQualityFilter,
    TemporalVerifier, ToxicityAuditor,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "dee",
    version,
    about = "Discrete Evidence Engine (DEE) — Deterministic, Non-Neural Data Integrity & Discovery Toolkit"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile a CSV dataset without modifying it.
    Inspect {
        input: PathBuf,
        /// Optional JSON report output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// List numeric observations that are marginally unusual under robust MAD statistics.
    Investigate {
        input: PathBuf,
        /// Robust-z magnitude threshold. Default: 3.5.
        #[arg(long, default_value_t = 3.5)]
        threshold: f64,
    },
    /// Export an inferred structural schema contract from a CSV dataset.
    ExportSchema {
        input: PathBuf,
        /// Destination path for generated schema JSON file.
        #[arg(short, long)]
        output: PathBuf,
        /// Optional contract name (defaults to input file stem).
        #[arg(long)]
        name: Option<String>,
    },
    /// Validate a CSV dataset against a declared Schema Contract.
    Validate {
        input: PathBuf,
        /// Path to Schema Contract JSON file.
        #[arg(short, long)]
        schema: PathBuf,
        /// Optional JSON validation report output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Generate an Evidence Ledger recording discrete audit assessments and investigation states.
    Ledger {
        input: PathBuf,
        /// Optional path to Schema Contract JSON file for contract verification.
        #[arg(short, long)]
        schema: Option<PathBuf>,
        /// Robust-z magnitude threshold for statistical rarity investigation. Default: 3.5.
        #[arg(long, default_value_t = 3.5)]
        threshold: f64,
        /// Optional JSON ledger output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Independently evaluate raw text quality, entropy, repetition, and anomalies without external metadata.
    TextProfile {
        input: PathBuf,
        /// Optional name of the text column (auto-detected if omitted).
        #[arg(short, long)]
        column: Option<String>,
        /// Robust-z threshold for text anomaly detection. Default: 3.5.
        #[arg(long, default_value_t = 3.5)]
        threshold: f64,
        /// Optional JSON report output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Filter a dataset systematically: purges HTML/image dumps, troll/conspiracies, and grammatical collapse.
    Filter {
        input: PathBuf,
        /// Destination clean output file (CSV or JSONL).
        #[arg(short, long)]
        output: PathBuf,
        /// Text column name (auto-detected if omitted).
        #[arg(short, long)]
        column: Option<String>,
        /// Minimum words per document. Default: 30.
        #[arg(long, default_value_t = 30)]
        min_words: usize,
        /// Maximum words per document. Default: 10000.
        #[arg(long, default_value_t = 10000)]
        max_words: usize,
        /// Minimum grammar score in [0.0, 1.0]. Default: 0.50.
        #[arg(long, default_value_t = 0.50)]
        min_grammar: f64,
        /// Disable purging unscientific conspiracies and flat earth claims.
        #[arg(long)]
        allow_conspiracies: bool,
    },
    /// Deterministically verify factual integrity of text (relational manifold energy, Harary spectral frustration, MDL complexity).
    VerifyFacts {
        /// Text string to evaluate directly or path to a text file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Near-deduplicate a dataset using MinHash LSH (128 permutation hashes, banded Jaccard).
    Dedup {
        input: PathBuf,
        /// Destination deduplicated output file.
        #[arg(short, long)]
        output: PathBuf,
        /// Text column name (auto-detected if omitted).
        #[arg(short, long)]
        column: Option<String>,
        /// Jaccard similarity threshold in [0.0, 1.0]. Default: 0.80.
        #[arg(long, default_value_t = 0.80)]
        threshold: f64,
    },
    /// Evaluate pedagogical reading level and SmolLM2 curriculum tier (FKGL, Dale-Chall, AWL density).
    Grade {
        /// Text string to evaluate directly or path to a text file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Verify temporal consistency and anachronism contradictions using Allen's Interval Algebra.
    VerifyTemporal {
        /// Text string to evaluate directly or path to a text file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Verify LaTeX mathematical delimiter balance, environment scopes, and OCR formula integrity.
    VerifyMath {
        /// Text string to evaluate directly or path to a text file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Verify code syntax, bracket balance, comment ratio, and detect minified code bundles.
    VerifyCode {
        /// Code string to evaluate directly or path to a file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Verify geopolitical balance, perspective attribution, and detect dogmatic nationalist propaganda.
    VerifyGeopolitics {
        /// Text string to evaluate directly or path to a file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Audit text for dehumanization vectors, demographic pestilence projection, and hate speech.
    AuditToxicity {
        /// Text string to evaluate directly or path to a file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Verify adherence to physical conservation laws (thermodynamics, Avogadro limit) and detect quackery.
    VerifyPseudoscience {
        /// Text string to evaluate directly or path to a file.
        input: String,
        /// Optional JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run the full multi-threaded pipeline: Rayon ingestion, LSH dedup, quality filter, and SmolLM2 curriculum partitioning.
    Pipeline {
        input: PathBuf,
        /// Destination directory for partitioned curriculum datasets (stage1 to stage4).
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Text column name (auto-detected if omitted).
        #[arg(short, long)]
        column: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Inspect { input, json } => match inspect_csv(&input) {
            Ok(report) => {
                if let Some(path) = json {
                    match report.to_json_file(&path) {
                        Ok(()) => println!("JSON report written: {}", path.display()),
                        Err(e) => eprintln!("Failed writing JSON report: {e}"),
                    }
                }
                println!("{}", report.to_pretty_text());
                Ok(())
            }
            Err(e) => Err(e),
        },
        Commands::Investigate { input, threshold } => {
            if !threshold.is_finite() || threshold <= 0.0 {
                Err("threshold must be a finite positive number".to_string())
            } else {
                investigate_csv(&input, threshold).map(|items| {
                    if items.is_empty() {
                        println!("No marginal rarity candidates found at robust-z >= {threshold}.");
                    } else {
                        println!("Marginal rarity candidates (NOT corruption verdicts):");
                        for item in items {
                            println!(
                                "row={} column={} value={} robust_z={:.4}",
                                item.row_index, item.column, item.value, item.robust_z
                            );
                        }
                    }
                })
            }
        }
        Commands::ExportSchema { input, output, name } => {
            let contract_name = name.unwrap_or_else(|| {
                input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("dataset_schema")
                    .to_string()
            });
            match infer_schema_from_csv(&input, Some(&contract_name)) {
                Ok(schema) => match schema.to_json_file(&output) {
                    Ok(()) => {
                        println!(
                            "Schema Contract '{}' successfully exported to {}",
                            contract_name,
                            output.display()
                        );
                        Ok(())
                    }
                    Err(e) => Err(format!("Failed to write schema JSON: {e}")),
                },
                Err(e) => Err(format!("Failed to infer schema: {e}")),
            }
        }
        Commands::Validate {
            input,
            schema,
            json,
        } => match SchemaContract::from_json_file(&schema) {
            Ok(contract) => match validate_csv(&input, &contract) {
                Ok(report) => {
                    if let Some(path) = json {
                        if let Err(e) = report.to_json_file(&path) {
                            eprintln!("Failed writing JSON validation report: {e}");
                        } else {
                            println!("JSON validation report written: {}", path.display());
                        }
                    }
                    let _ = std::io::Write::write_all(
                        &mut std::io::stdout(),
                        report.to_pretty_text().as_bytes(),
                    );
                    if report.is_valid {
                        Ok(())
                    } else {
                        Err(format!(
                            "Validation failed with {} contract violation(s)",
                            report.violations.len()
                        ))
                    }
                }
                Err(e) => Err(format!("Validation execution error: {e}")),
            },
            Err(e) => Err(format!("Failed to load schema contract: {e}")),
        },
        Commands::Ledger {
            input,
            schema,
            threshold,
            json,
        } => {
            let loaded_schema = match schema {
                Some(ref p) => match SchemaContract::from_json_file(p) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        eprintln!("Error loading schema file: {e}");
                        std::process::exit(1);
                    }
                },
                None => None,
            };

            match generate_evidence_ledger(&input, loaded_schema.as_ref(), threshold) {
                Ok(ledger) => {
                    if let Some(path) = json {
                        if let Err(e) = ledger.to_json_file(&path) {
                            eprintln!("Failed writing JSON evidence ledger: {e}");
                        } else {
                            println!("JSON evidence ledger written: {}", path.display());
                        }
                    }
                    let _ = std::io::Write::write_all(
                        &mut std::io::stdout(),
                        ledger.to_pretty_text().as_bytes(),
                    );
                    Ok(())
                }
                Err(e) => Err(format!("Failed generating evidence ledger: {e}")),
            }
        }
        Commands::TextProfile {
            input,
            column,
            threshold,
            json,
        } => {
            if !threshold.is_finite() || threshold <= 0.0 {
                return eprintln!("error: threshold must be a finite positive number");
            }
            match evaluate_text_column(&input, column.as_deref(), threshold) {
                Ok(report) => {
                    if let Some(path) = json {
                        if let Err(e) = report.to_json_file(&path) {
                            eprintln!("Failed writing JSON text quality report: {e}");
                        } else {
                            println!("JSON text quality report written: {}", path.display());
                        }
                    }
                    let _ = std::io::Write::write_all(
                        &mut std::io::stdout(),
                        report.to_pretty_text().as_bytes(),
                    );
                    Ok(())
                }
                Err(e) => Err(format!("Text evaluation error: {e}")),
            }
        }
        Commands::Filter {
            input,
            output,
            column,
            min_words,
            max_words,
            min_grammar,
            allow_conspiracies,
        } => {
            let filter = SystematicQualityFilter {
                min_words,
                max_words,
                min_grammar_score: min_grammar,
                purge_conspiracies: !allow_conspiracies,
                purge_html_and_images: true,
                purge_boilerplate: true,
                epistemic_engine: dee::EpistemicMarkovEngine::default(),
                factual_verifier: dee::FactualVerifier::default(),
                temporal_verifier: dee::TemporalVerifier::default(),
            };
            println!("Applying DEE Systematic Quality Filter to {}...", input.display());
            match filter_dataset(&input, &output, column.as_deref(), &filter) {
                Ok(summary) => {
                    println!("================================================================================");
                    println!("DEE SYSTEMATIC FILTER COMPLETE");
                    println!("Output File: {}", output.display());
                    println!(
                        "Total Input Documents: {} | Clean Kept: {} ({:.1}%) | Purged: {} ({:.1}%)",
                        summary.total_input_documents,
                        summary.kept_clean_documents,
                        summary.clean_token_retention_ratio * 100.0,
                        summary.purged_documents,
                        (1.0 - summary.clean_token_retention_ratio) * 100.0
                    );
                    println!("Purged Breakdown by Defect:");
                    for (reason, count) in &summary.purged_by_reason {
                        println!("  - {:<35}: {} docs", reason, count);
                    }
                    println!("================================================================================");
                    Ok(())
                }
                Err(e) => Err(format!("Filtering failed: {e}")),
            }
        }
        Commands::VerifyFacts { input, json } => {
            let text = if std::path::Path::new(&input).exists() {
                match std::fs::read_to_string(&input) {
                    Ok(content) => content,
                    Err(e) => return eprintln!("Error reading file {input}: {e}"),
                }
            } else {
                input
            };

            let verifier = dee::FactualVerifier::default();
            let result = verifier.verify_text(&text);

            if let Some(path) = json {
                match serde_json::to_string_pretty(&result) {
                    Ok(json_str) => {
                        if let Err(e) = std::fs::write(&path, json_str) {
                            eprintln!("Failed writing JSON factual report: {e}");
                        } else {
                            println!("JSON factual verification report written: {}", path.display());
                        }
                    }
                    Err(e) => eprintln!("Serialization error: {e}"),
                }
            }

            println!("================================================================================");
            println!("DEE DETERMINISTIC FACTUAL VERIFICATION REPORT");
            println!("Consistent: {}", if result.is_factually_consistent { "YES (PASSED)" } else { "NO (VIOLATION DETECTED)" });
            println!("Summary:    {}", result.summary);
            println!("--------------------------------------------------------------------------------");
            println!("Mathematical Diagnostics:");
            println!("  - RotatE Manifold Relational Energy:  {:.4}", result.total_manifold_energy);
            println!("  - Harary Signed Frustration \u{03bb}_1(\u{44c}):     {:.4}", result.spectral_frustration_lambda1);
            println!("  - Harary Unbalanced Negative Cycles:  {}", result.unbalanced_cycles);
            println!("  - MDL Algorithmic Excess Complexity:  {:.1} bits", result.mdl_excess_bits);
            if !result.violations.is_empty() {
                println!("--------------------------------------------------------------------------------");
                println!("Detected Factual Violations ({}):", result.violations.len());
                for (i, v) in result.violations.iter().enumerate() {
                    println!("  [{}] Category: {}", i + 1, v.category);
                    println!("      Detail:   {}", v.explanation);
                }
            }
            println!("================================================================================");
            Ok(())
        }
        Commands::Dedup {
            input,
            output,
            column,
            threshold,
        } => (|| -> Result<(), String> {
            let config = MinHashLshConfig {
                jaccard_threshold: threshold,
                ..Default::default()
            };
            let engine = MinHashLshEngine::new(config);
            println!("Reading dataset for MinHash LSH deduplication from {}...", input.display());

            let file = std::fs::File::open(&input).map_err(|e| format!("Open error: {e}"))?;
            let rdr = std::io::BufReader::new(file);
            let mut csv_rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(rdr);
            let headers = csv_rdr.headers().map_err(|e| format!("Header error: {e}"))?.clone();
            let col_idx = if let Some(target) = column.as_deref() {
                headers.iter().position(|h| h == target).ok_or_else(|| format!("Target column '{target}' not found"))?
            } else {
                headers.iter().position(|h| {
                    let l = h.to_lowercase();
                    l == "text" || l == "content" || l == "body"
                }).unwrap_or(0)
            };

            let mut docs = Vec::new();
            let mut records = Vec::new();
            for rec in csv_rdr.records() {
                if let Ok(r) = rec {
                    if let Some(val) = r.get(col_idx) {
                        docs.push(val.to_string());
                        records.push(r);
                    }
                }
            }

            println!("Running MinHash LSH deduplication across {} documents...", docs.len());
            let doc_slices: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            let (mask, summary) = engine.deduplicate_texts(&doc_slices);

            let out_file = std::fs::File::create(&output).map_err(|e| format!("Create error: {e}"))?;
            let mut wtr = csv::Writer::from_writer(out_file);
            wtr.write_record(&headers).map_err(|e| format!("Write header: {e}"))?;
            for (i, r) in records.iter().enumerate() {
                if mask[i] {
                    wtr.write_record(r).map_err(|e| format!("Write record: {e}"))?;
                }
            }
            wtr.flush().map_err(|e| format!("Flush error: {e}"))?;

            println!("================================================================================");
            println!("MINHASH LSH DEDUPLICATION COMPLETE");
            println!("Output File: {}", output.display());
            println!(
                "Total: {} | Unique Kept: {} ({:.1}%) | Duplicates Purged: {} ({:.1}%)",
                summary.total_documents,
                summary.unique_documents_kept,
                summary.retention_ratio * 100.0,
                summary.duplicate_documents_purged,
                (1.0 - summary.retention_ratio) * 100.0
            );
            println!("Duplicate Clusters Detected: {}", summary.duplicate_clusters_count);
            println!("================================================================================");
            Ok(())
        })(),
        Commands::Grade { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let scorer = PedagogicalScorer::default();
            let report = scorer.score_document(&text);

            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON pedagogical report written: {}", path.display());
                }
            }

            println!("================================================================================");
            println!("DEE DETERMINISTIC PEDAGOGICAL & READABILITY REPORT");
            println!("Curriculum Tier:   {:?}", report.recommended_stage);
            println!("Summary:           {}", report.pedagogical_summary);
            println!("--------------------------------------------------------------------------------");
            println!("Readability Metrics:");
            println!("  - Flesch-Kincaid Grade Level (FKGL):  {:.1}", report.fkgl_grade_level);
            println!("  - Flesch Reading Ease (FRE):          {:.1}", report.flesch_reading_ease);
            println!("  - Dale-Chall Readability Index:       {:.2}", report.dale_chall_score);
            println!("  - Academic Word List (AWL) Density:   {:.2}%", report.academic_word_density * 100.0);
            println!("  - Complex Words Ratio (>= 3 syll):    {:.2}%", report.complex_words_ratio * 100.0);
            println!("  - Counts: {} words, {} sentences, {} syllables", report.words_count, report.sentences_count, report.syllables_count);
            println!("================================================================================");
            Ok(())
        })(),
        Commands::VerifyTemporal { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let verifier = TemporalVerifier::default();
            let report = verifier.verify_text(&text);

            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON temporal report written: {}", path.display());
                }
            }

            println!("================================================================================");
            println!("DEE TEMPORAL INTERVAL CONSISTENCY REPORT (ALLEN'S ALGEBRA)");
            println!("Consistent: {}", if report.is_temporally_consistent { "YES (PASSED)" } else { "NO (ANACHRONISM DETECTED)" });
            println!("Summary:    {}", report.summary);
            if !report.violations.is_empty() {
                println!("--------------------------------------------------------------------------------");
                println!("Anachronisms Detected ({}):", report.violations.len());
                for (i, v) in report.violations.iter().enumerate() {
                    println!("  [{}] Relation: {:?} | {}", i + 1, v.computed_relation, v.explanation);
                }
            }
            println!("================================================================================");
            Ok(())
        })(),
        Commands::VerifyMath { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let report = MathValidator::validate_math_text(&text);

            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON math report written: {}", path.display());
                }
            }

            println!("================================================================================");
            println!("DEE MATHEMATICAL & FORMULA INTEGRITY REPORT");
            println!("Valid:      {}", if report.is_valid { "YES (PASSED)" } else { "NO (SYNTAX/OCR DEFECTS)" });
            println!("Math Blocks: {} inline, {} display", report.inline_math_blocks, report.display_math_blocks);
            println!("OCR Junk Ratio: {:.2}%", report.ocr_garbage_ratio * 100.0);
            if !report.defect_explanations.is_empty() {
                println!("--------------------------------------------------------------------------------");
                println!("Defects ({}):", report.defect_explanations.len());
                for (i, d) in report.defect_explanations.iter().enumerate() {
                    println!("  [{}] {}", i + 1, d);
                }
            }
            println!("================================================================================");
            Ok(())
        })(),
        Commands::VerifyCode { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let report = CodeValidator::validate_code(&text);

            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON code report written: {}", path.display());
                }
            }

            println!("================================================================================");
            println!("DEE CODE AST & INTEGRITY REPORT");
            println!("Valid:      {}", if report.is_valid { "YES (PASSED)" } else { "NO (SYNTAX/DELIMITER DEFECTS)" });
            println!("Language:   {}", report.detected_language.as_deref().unwrap_or("generic"));
            println!("Lines:      {} total, {} code, {} comment ({:.1}% comment ratio)", report.total_lines, report.code_lines, report.comment_lines, report.comment_ratio * 100.0);
            println!("Minified:   {}", if report.is_minified_bundle { "YES (PURGED)" } else { "NO" });
            println!("Truncated:  {}", if report.is_truncated { "YES (PURGED)" } else { "NO" });
            if !report.syntax_defects.is_empty() {
                println!("--------------------------------------------------------------------------------");
                println!("Defects ({}):", report.syntax_defects.len());
                for (i, d) in report.syntax_defects.iter().enumerate() {
                    println!("  [{}] {}", i + 1, d);
                }
            }
            println!("================================================================================");
            Ok(())
        })(),
        Commands::VerifyGeopolitics { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let rep = GeopoliticalBalancer::evaluate_text(&text);
            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&rep) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON geopolitical report written: {}", path.display());
                }
            }
            println!("================================================================================");
            println!("DEE GEOPOLITICAL PERSPECTIVE ATTRIBUTION REPORT");
            println!("Balanced / Attributed: {}", if rep.is_balanced_or_attributed { "YES (PASSED)" } else { "NO (DOGMATIC PROPAGANDA)" });
            println!("Attribution Score:     {:.2}", rep.attribution_score);
            println!("Disputed Regions:      {:?}", rep.disputed_regions_detected);
            println!("Summary:               {}", rep.explanation);
            println!("================================================================================");
            Ok(())
        })(),
        Commands::AuditToxicity { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let rep = ToxicityAuditor::audit_text(&text);
            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&rep) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON toxicity report written: {}", path.display());
                }
            }
            println!("================================================================================");
            println!("DEE TOXICITY & DEHUMANIZATION AUDIT REPORT");
            println!("Clean:                 {}", if rep.is_clean { "YES (PASSED)" } else { "NO (DEHUMANIZATION / HATE DETECTED)" });
            println!("Dehumanization Score:  {:.2}", rep.dehumanization_score);
            println!("Demographics Matched:  {:?}", rep.detected_demographics);
            if !rep.biological_pestilence_matches.is_empty() {
                println!("Pestilence Vectors:    {:?}", rep.biological_pestilence_matches);
            }
            if !rep.stereotypic_essentialism_matches.is_empty() {
                println!("Essentialist Claims:   {:?}", rep.stereotypic_essentialism_matches);
            }
            println!("Summary:               {}", rep.explanation);
            println!("================================================================================");
            Ok(())
        })(),
        Commands::VerifyPseudoscience { input, json } => (|| -> Result<(), String> {
            let text = if std::path::Path::new(&input).exists() {
                std::fs::read_to_string(&input).map_err(|e| format!("Read file: {e}"))?
            } else {
                input
            };
            let rep = PseudoscienceFilter::evaluate_text(&text);
            if let Some(path) = json {
                if let Ok(json_str) = serde_json::to_string_pretty(&rep) {
                    let _ = std::fs::write(&path, json_str);
                    println!("JSON pseudoscience report written: {}", path.display());
                }
            }
            println!("================================================================================");
            println!("DEE CONSERVATION LAW & PSEUDOSCIENCE REPORT");
            println!("Scientifically Grounded: {}", if rep.is_scientifically_grounded { "YES (PASSED)" } else { "NO (QUACKERY / CONSERVATION VIOLATION)" });
            println!("Debunking Refutation:    {}", if rep.is_debunking_refutation { "YES" } else { "NO" });
            println!("Summary:                 {}", rep.explanation);
            if !rep.thermodynamic_violations.is_empty() {
                println!("Thermodynamic Violations ({}):", rep.thermodynamic_violations.len());
                for (i, v) in rep.thermodynamic_violations.iter().enumerate() {
                    println!("  [{}] {}", i + 1, v);
                }
            }
            if !rep.avogadro_limit_violations.is_empty() {
                println!("Avogadro Limit Violations ({}):", rep.avogadro_limit_violations.len());
                for (i, v) in rep.avogadro_limit_violations.iter().enumerate() {
                    println!("  [{}] {}", i + 1, v);
                }
            }
            if !rep.quackery_cures_detected.is_empty() {
                println!("Quackery Assertions ({}):", rep.quackery_cures_detected.len());
                for (i, v) in rep.quackery_cures_detected.iter().enumerate() {
                    println!("  [{}] {}", i + 1, v);
                }
            }
            println!("================================================================================");
            Ok(())
        })(),
        Commands::Pipeline {
            input,
            output_dir,
            column,
        } => {
            println!("Initializing DEE Multi-Threaded Parallel Pipeline...");
            let pipeline = ParallelPipeline::default();
            match pipeline.process_dataset(&input, &output_dir, column.as_deref()) {
                Ok(summary) => {
                    println!("================================================================================");
                    println!("DEE FULL PIPELINE EXECUTION COMPLETE");
                    println!("Input File:        {}", input.display());
                    println!("Output Directory:  {}", output_dir.display());
                    println!(
                        "Throughput:        {:.1} docs/sec (Total: {} docs in {:.2}s)",
                        summary.throughput_docs_per_sec,
                        summary.total_input_documents,
                        summary.elapsed_seconds
                    );
                    println!("--------------------------------------------------------------------------------");
                    println!("Deduplication (MinHash LSH):");
                    println!("  - Unique Kept:       {} docs", summary.deduplicated_documents);
                    println!("  - Duplicates Purged: {} docs", summary.duplicates_purged);
                    println!("Quality & Epistemic Filter:");
                    println!("  - Clean Kept:        {} docs", summary.quality_clean_documents);
                    println!("  - Defects Purged:    {} docs", summary.quality_defects_purged);
                    println!("SmolLM2 Curriculum Partitioning:");
                    println!("  - Stage 1 (Fundamental):   {} docs -> stage1_fundamental", summary.stage1_count);
                    println!("  - Stage 2 (Intermediate):  {} docs -> stage2_intermediate", summary.stage2_count);
                    println!("  - Stage 3 (Advanced):      {} docs -> stage3_advanced", summary.stage3_count);
                    println!("  - Stage 4 (Research):      {} docs -> stage4_research", summary.stage4_count);
                    println!("================================================================================");
                    Ok(())
                }
                Err(e) => Err(format!("Pipeline execution failed: {e}")),
            }
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}
