use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::Path;

use crate::code_validator::CodeValidator;
use crate::epistemic_markov::EpistemicMarkovEngine;
use crate::factual_verifier::FactualVerifier;
use crate::geopolitical_balancer::GeopoliticalBalancer;
use crate::math_validator::MathValidator;
use crate::pseudoscience_filter::PseudoscienceFilter;
use crate::temporal_verifier::TemporalVerifier;
use crate::toxicity_auditor::ToxicityAuditor;

/// Categories of systematic data purging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PurgeReason {
    HtmlAndMarkupDump,
    ImageAndMediaTagDump,
    UnscientificConspiracyOrTroll,
    FactualContradictionOrConspiracy,
    TemporalAnachronismContradiction,
    MathematicalSyntaxOrOcrCollapse,
    CodeSyntaxOrMinifiedCollapse,
    DogmaticGeopoliticalPropaganda,
    RacialDiscriminationOrHate,
    PseudoscienceAndConservationViolation,
    PseudoscienceQuackery,
    GrammarAndSyntacticCollapse,
    RepetitiveBoilerplateOrNavigation,
    LogOrCodeDump,
    ContextTooFragmented,
}

impl PurgeReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HtmlAndMarkupDump => "html_markup_dump",
            Self::ImageAndMediaTagDump => "image_media_tag_dump",
            Self::UnscientificConspiracyOrTroll => "conspiracy_troll_claim",
            Self::FactualContradictionOrConspiracy => "factual_contradiction_or_conspiracy",
            Self::TemporalAnachronismContradiction => "temporal_anachronism_contradiction",
            Self::MathematicalSyntaxOrOcrCollapse => "math_syntax_ocr_collapse",
            Self::CodeSyntaxOrMinifiedCollapse => "code_syntax_minified_collapse",
            Self::DogmaticGeopoliticalPropaganda => "dogmatic_geopolitical_propaganda",
            Self::RacialDiscriminationOrHate => "racial_discrimination_or_hate",
            Self::PseudoscienceAndConservationViolation => "pseudoscience_conservation_violation",
            Self::PseudoscienceQuackery => "pseudoscience_quackery",
            Self::GrammarAndSyntacticCollapse => "grammar_syntactic_collapse",
            Self::RepetitiveBoilerplateOrNavigation => "boilerplate_navigation_spam",
            Self::LogOrCodeDump => "log_or_code_dump",
            Self::ContextTooFragmented => "context_too_fragmented",
        }
    }
}

/// Detailed evaluation score for a single document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentQualityScore {
    pub grammar_score: f64,
    pub context_score: f64,
    pub html_artifact_count: usize,
    pub image_artifact_count: usize,
    pub conspiracy_matches: Vec<String>,
    pub is_clean: bool,
    pub purge_reasons: Vec<PurgeReason>,
    pub explanation: String,
}

/// Fast deterministic grammar and syntactic coherence analyzer.
pub struct GrammarScorer;

impl GrammarScorer {
    /// Computes a grammar & syntactic coherence score in [0.0, 1.0].
    /// Higher is cleaner, well-structured natural language.
    pub fn score(text: &str) -> (f64, Vec<String>) {
        let mut defects = Vec::new();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return (0.0, vec!["Empty text".to_string()]);
        }

        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.is_empty() {
            return (0.0, vec!["No words".to_string()]);
        }

        let sentences: Vec<&str> = trimmed
            .split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            return (0.1, vec!["No sentence structure".to_string()]);
        }

        let mut capitalization_failures = 0usize;
        for s in &sentences {
            if let Some(first_char) = s.chars().next() {
                if first_char.is_alphabetic() && !first_char.is_uppercase() {
                    capitalization_failures += 1;
                }
            }
        }
        let cap_ratio = 1.0 - (capitalization_failures as f64 / sentences.len() as f64);
        if cap_ratio < 0.6 {
            defects.push(format!("Frequent lowercase sentence starts ({:.0}%)", (1.0 - cap_ratio) * 100.0));
        }

        // 2. Check for duplicate adjacent words ("the the", "is is")
        let mut duplicate_adjacent = 0usize;
        for i in 1..words.len() {
            let prev = words[i - 1].trim_matches(|c: char| c.is_ascii_punctuation()).to_ascii_lowercase();
            let curr = words[i].trim_matches(|c: char| c.is_ascii_punctuation()).to_ascii_lowercase();
            if !prev.is_empty() && prev == curr && prev.len() > 1 {
                duplicate_adjacent += 1;
            }
        }
        let dup_penalty = (duplicate_adjacent as f64 / words.len() as f64) * 10.0;
        if duplicate_adjacent > 2 {
            defects.push(format!("Found {} duplicate adjacent word pairs", duplicate_adjacent));
        }

        // 3. Punctuation spacing defects (e.g. "word ,word", "word.Word", "word !word")
        let mut bad_spacing = 0usize;
        let bytes = trimmed.as_bytes();
        for i in 1..bytes.len() {
            let b = bytes[i];
            let prev = bytes[i - 1];
            // Space before comma or period (e.g. "foo ,")
            if (b == b',' || b == b'.' || b == b'!' || b == b'?') && prev == b' ' {
                bad_spacing += 1;
            }
            // Missing space after sentence punctuation ("foo.Bar")
            if (prev == b'.' || prev == b'!' || prev == b'?') && b.is_ascii_uppercase() {
                bad_spacing += 1;
            }
        }
        let space_penalty = (bad_spacing as f64 / words.len() as f64).min(0.4);
        if bad_spacing > 5 {
            defects.push(format!("Punctuation spacing errors: {}", bad_spacing));
        }

        // 4. Mean sentence length coherence (penalize extreme run-ons or fragmented single-word sentences)
        let avg_sentence_len = words.len() as f64 / sentences.len() as f64;
        let mut sentence_len_penalty = 0.0;
        if avg_sentence_len < 3.0 {
            sentence_len_penalty = 0.35;
            defects.push("Excessively fragmented sentences (mean < 3 words)".to_string());
        } else if avg_sentence_len > 60.0 {
            sentence_len_penalty = 0.25;
            defects.push("Severe run-on sentences (mean > 60 words)".to_string());
        }

        // 5. Letter case shouting / chaos (e.g. "ALL CAPS" or "lIkE tHiS")
        let alpha_chars: Vec<char> = trimmed.chars().filter(|c| c.is_alphabetic()).collect();
        let upper_ratio = if !alpha_chars.is_empty() {
            alpha_chars.iter().filter(|c| c.is_uppercase()).count() as f64 / alpha_chars.len() as f64
        } else {
            0.0
        };
        let mut case_penalty = 0.0;
        if upper_ratio > 0.65 && alpha_chars.len() > 40 {
            case_penalty = 0.40;
            defects.push("Excessive uppercase shouting (> 65% caps)".to_string());
        }

        let base_score = 1.0 - (1.0 - cap_ratio) * 0.3 - dup_penalty.min(0.3) - space_penalty - sentence_len_penalty - case_penalty;
        let final_score = base_score.clamp(0.0, 1.0);
        (final_score, defects)
    }
}

// Systematic data dump patterns (raw HTML, image tags, CSS, boilerplate).
const HTML_IMAGE_MEDIA_PATTERNS: &[&str] = &[
    "<script", "</script>", "<style", "</style>",
    "<img ", "<img\n", "<img\t", "<image", "src=\"data:image",
    "![](", "![image](", "![photo](", "![fig](",
    "<div", "</div>", "<span", "</span>", "<table", "</table>",
    "<iframe", "</iframe>", "<video", "<audio", "href=\"javascript:",
    "class=\"nav", "class=\"header", "class=\"footer", "class=\"menu",
];

/// Systematic boilerplate patterns (web scrapers capturing cookie notices / terms).
const BOILERPLATE_PATTERNS: &[&str] = &[
    "all rights reserved",
    "cookie policy",
    "we use cookies to ensure you get the best experience",
    "terms of service apply",
    "subscribe to our newsletter",
    "click here to subscribe",
    "leave a comment cancel reply",
    "your email address will not be published",
];

pub struct SystematicQualityFilter {
    pub min_words: usize,
    pub max_words: usize,
    pub min_grammar_score: f64,
    pub purge_conspiracies: bool,
    pub purge_html_and_images: bool,
    pub purge_boilerplate: bool,
    pub epistemic_engine: EpistemicMarkovEngine,
    pub factual_verifier: FactualVerifier,
    pub temporal_verifier: TemporalVerifier,
}

impl Default for SystematicQualityFilter {
    fn default() -> Self {
        Self {
            min_words: 30,
            max_words: 10000,
            min_grammar_score: 0.50,
            purge_conspiracies: true,
            purge_html_and_images: true,
            purge_boilerplate: true,
            epistemic_engine: EpistemicMarkovEngine::default(),
            factual_verifier: FactualVerifier::default(),
            temporal_verifier: TemporalVerifier::default(),
        }
    }
}

impl SystematicQualityFilter {
    pub fn evaluate_document(&self, text: &str) -> DocumentQualityScore {
        let mut purge_reasons = Vec::new();
        let mut explanations = Vec::new();
        let lower = text.to_lowercase();
        let words: Vec<&str> = text.split_whitespace().collect();

        // 1. Context & Length Check
        if words.len() < self.min_words {
            purge_reasons.push(PurgeReason::ContextTooFragmented);
            explanations.push(format!("Context too brief ({} words < minimum {})", words.len(), self.min_words));
        } else if words.len() > self.max_words {
            purge_reasons.push(PurgeReason::ContextTooFragmented);
            explanations.push(format!("Context excessive ({} words > maximum {})", words.len(), self.max_words));
        }

        // 2. HTML & Image Tag Dump Check
        let mut html_count = 0usize;
        let mut image_count = 0usize;
        for pat in HTML_IMAGE_MEDIA_PATTERNS {
            let occurrences = lower.matches(pat).count();
            if pat.contains("img") || pat.contains("video") || pat.contains("audio") || pat.starts_with("![") {
                image_count += occurrences;
            } else {
                html_count += occurrences;
            }
        }

        if self.purge_html_and_images {
            if html_count >= 3 {
                purge_reasons.push(PurgeReason::HtmlAndMarkupDump);
                explanations.push(format!("Contains {} raw HTML tags/elements", html_count));
            }
            if image_count >= 2 {
                purge_reasons.push(PurgeReason::ImageAndMediaTagDump);
                explanations.push(format!("Contains {} image/media embedding tags", image_count));
            }
        }

        // 3. Epistemic Conductance & Graph Islanding (Markovian + Random Walk Diffusion)
        let mut conspiracy_matches = Vec::new();
        if self.purge_conspiracies {
            let epistemic_eval = self.epistemic_engine.evaluate_text(text);
            if epistemic_eval.is_epistemic_island {
                purge_reasons.push(PurgeReason::UnscientificConspiracyOrTroll);
                conspiracy_matches.push(format!(
                    "Epistemic Islanding: Conductance \u{03a6}={:.3} < 0.15",
                    epistemic_eval.epistemic_conductance
                ));
                explanations.push(epistemic_eval.explanation);
            } else if !epistemic_eval.is_syntactically_coherent {
                purge_reasons.push(PurgeReason::GrammarAndSyntacticCollapse);
                explanations.push(epistemic_eval.explanation);
            }

            // 3b. Deterministic Factual Verification (Manifold Energy, Harary Balance, MDL Complexity)
            let factual_eval = self.factual_verifier.verify_text(text);
            if !factual_eval.is_factually_consistent {
                purge_reasons.push(PurgeReason::FactualContradictionOrConspiracy);
                for v in &factual_eval.violations {
                    conspiracy_matches.push(format!("{}: {}", v.category, v.explanation));
                }
                explanations.push(factual_eval.summary);
            }

            // 3c. Temporal Consistency & Anachronism Detection (Allen's Interval Algebra)
            let temporal_eval = self.temporal_verifier.verify_text(text);
            if !temporal_eval.is_temporally_consistent {
                purge_reasons.push(PurgeReason::TemporalAnachronismContradiction);
                for v in &temporal_eval.violations {
                    explanations.push(v.explanation.clone());
                }
            }
        }

        // 3d. Mathematical Delimiter & Formula Integrity
        let math_eval = MathValidator::validate_math_text(text);
        if !math_eval.is_valid {
            purge_reasons.push(PurgeReason::MathematicalSyntaxOrOcrCollapse);
            explanations.push(format!("Math integrity errors: {}", math_eval.defect_explanations.join("; ")));
        }

        // 3e. Code Block Structural Integrity
        if text.contains("```") || text.starts_with("def ") || text.starts_with("fn ") {
            let code_eval = CodeValidator::validate_code(text);
            if !code_eval.is_valid {
                purge_reasons.push(PurgeReason::CodeSyntaxOrMinifiedCollapse);
                explanations.push(format!("Code integrity errors: {}", code_eval.syntax_defects.join("; ")));
            }
        }

        // 3f. Toxicity, Dehumanization Vectors & Hate Speech
        let toxicity_eval = ToxicityAuditor::audit_text(text);
        if !toxicity_eval.is_clean {
            purge_reasons.push(PurgeReason::RacialDiscriminationOrHate);
            explanations.push(toxicity_eval.explanation);
        }

        // 3g. Geopolitical Neutrality vs. Dogmatic Delegitimizing Propaganda
        let geopolitics_eval = GeopoliticalBalancer::evaluate_text(text);
        if !geopolitics_eval.is_balanced_or_attributed {
            purge_reasons.push(PurgeReason::DogmaticGeopoliticalPropaganda);
            explanations.push(geopolitics_eval.explanation);
        }

        // 3h. Pseudoscience, Quackery & Conservation Law Violations
        let pseudoscience_eval = PseudoscienceFilter::evaluate_text(text);
        if !pseudoscience_eval.is_scientifically_grounded {
            purge_reasons.push(PurgeReason::PseudoscienceAndConservationViolation);
            explanations.push(pseudoscience_eval.explanation);
        }

        // 4. Repetitive Boilerplate / Scraped Footers
        if self.purge_boilerplate {
            let mut boilerplate_hits = 0usize;
            for bp in BOILERPLATE_PATTERNS {
                if lower.contains(bp) {
                    boilerplate_hits += 1;
                }
            }
            if boilerplate_hits >= 2 {
                purge_reasons.push(PurgeReason::RepetitiveBoilerplateOrNavigation);
                explanations.push(format!("Contains {} scraped website boilerplate / cookie notices", boilerplate_hits));
            }
        }

        // 5. Fast Grammar & Syntactic Coherence Scorer
        let (grammar_score, grammar_defects) = GrammarScorer::score(text);
        if grammar_score < self.min_grammar_score {
            purge_reasons.push(PurgeReason::GrammarAndSyntacticCollapse);
            explanations.push(format!(
                "Grammar score {:.2} below threshold {:.2} ({})",
                grammar_score,
                self.min_grammar_score,
                grammar_defects.join(", ")
            ));
        }

        let is_clean = purge_reasons.is_empty();
        let context_score = (words.len() as f64 / 500.0).clamp(0.1, 1.0);

        DocumentQualityScore {
            grammar_score,
            context_score,
            html_artifact_count: html_count,
            image_artifact_count: image_count,
            conspiracy_matches,
            is_clean,
            purge_reasons,
            explanation: if explanations.is_empty() {
                "Document satisfies all grammar, contextual, and epistemic criteria".to_string()
            } else {
                explanations.join(" | ")
            },
        }
    }
}

/// Statistics from running systematic filtering on a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSummary {
    pub total_input_documents: usize,
    pub kept_clean_documents: usize,
    pub purged_documents: usize,
    pub purged_by_reason: std::collections::HashMap<String, usize>,
    pub clean_token_retention_ratio: f64,
}

/// Executes systematic quality filtering on a dataset (CSV or JSONL), writing clean records to output.
pub fn filter_dataset(
    input_path: &Path,
    output_path: &Path,
    text_column_name: Option<&str>,
    filter: &SystematicQualityFilter,
) -> Result<FilterSummary, String> {
    let is_jsonl = input_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false);

    let mut total_docs = 0usize;
    let mut kept_docs = 0usize;
    let mut purged_docs = 0usize;
    let mut reason_counts = std::collections::HashMap::new();

    if is_jsonl {
        let in_file = File::open(input_path).map_err(|e| format!("JSONL read error: {e}"))?;
        let reader = BufReader::new(in_file);
        let mut out_file = File::create(output_path).map_err(|e| format!("Output JSONL write error: {e}"))?;

        let mut determined_col: Option<String> = text_column_name.map(|s| s.to_string());

        for (idx, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("JSONL line error at {idx}: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            total_docs += 1;

            let val: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("JSON parse error at line {}: {e}", idx + 1))?;

            let text_val = if let Some(ref col) = determined_col {
                crate::text_evaluator::extract_text_from_json_value(&val, col)
            } else {
                let candidates = [
                    "text", "content", "response", "output", "prompt", "instruction",
                    "document", "body", "messages",
                ];
                let mut found = None;
                for c in candidates {
                    if let Some(t) = crate::text_evaluator::extract_text_from_json_value(&val, c) {
                        determined_col = Some(c.to_string());
                        found = Some(t);
                        break;
                    }
                }
                found
            }
            .unwrap_or_default();

            let score = filter.evaluate_document(&text_val);
            if score.is_clean {
                kept_docs += 1;
                writeln!(out_file, "{}", trimmed).map_err(|e| format!("Write error: {e}"))?;
            } else {
                purged_docs += 1;
                for r in &score.purge_reasons {
                    *reason_counts.entry(r.as_str().to_string()).or_insert(0) += 1;
                }
            }
        }
    } else {
        let mut rdr = csv::Reader::from_path(input_path).map_err(|e| format!("CSV read error: {e}"))?;
        let headers = rdr.headers().map_err(|e| format!("CSV header error: {e}"))?.clone();

        let text_col_idx = if let Some(name) = text_column_name {
            headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or_else(|| format!("Column '{name}' not found in CSV"))?
        } else {
            let candidates = ["text", "content", "document", "body", "raw_text", "article"];
            let mut found = None;
            for c in candidates {
                if let Some(idx) = headers.iter().position(|h| h.eq_ignore_ascii_case(c)) {
                    found = Some(idx);
                    break;
                }
            }
            found.ok_or_else(|| "No text column found in CSV".to_string())?
        };

        let out_file = File::create(output_path).map_err(|e| format!("Output CSV write error: {e}"))?;
        let mut wtr = csv::Writer::from_writer(out_file);
        wtr.write_record(&headers).map_err(|e| format!("CSV header write error: {e}"))?;

        for record in rdr.records() {
            let record = record.map_err(|e| format!("CSV record error: {e}"))?;
            total_docs += 1;
            let text = record.get(text_col_idx).unwrap_or("");
            let score = filter.evaluate_document(text);

            if score.is_clean {
                kept_docs += 1;
                wtr.write_record(&record).map_err(|e| format!("CSV record write error: {e}"))?;
            } else {
                purged_docs += 1;
                for r in &score.purge_reasons {
                    *reason_counts.entry(r.as_str().to_string()).or_insert(0) += 1;
                }
            }
        }
        wtr.flush().map_err(|e| format!("CSV flush error: {e}"))?;
    }

    let retention = if total_docs > 0 {
        kept_docs as f64 / total_docs as f64
    } else {
        0.0
    };

    Ok(FilterSummary {
        total_input_documents: total_docs,
        kept_clean_documents: kept_docs,
        purged_documents: purged_docs,
        purged_by_reason: reason_counts,
        clean_token_retention_ratio: retention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_scoring_clean_prose() {
        let prose = "This is a clean, well-formed sentence about astrophysics. The planets orbit around the Sun according to Newtonian gravitation and general relativity.";
        let (score, defects) = GrammarScorer::score(prose);
        assert!(score >= 0.85, "Expected high grammar score for clean prose, got {}", score);
        assert!(defects.is_empty());
    }

    #[test]
    fn test_grammar_scoring_catches_degraded_text() {
        let bad = "word ,another word .no space before and lower case start .and then repeated repeated word word .";
        let (score, defects) = GrammarScorer::score(bad);
        assert!(score < 0.65, "Expected lower score for bad grammar, got {}", score);
        assert!(!defects.is_empty());
    }

    #[test]
    fn test_filter_purges_conspiracies_and_trolls() {
        let filter = SystematicQualityFilter::default();

        let flat_earth_doc = "Many people do not realize that the flatearth is completely real and surrounded by an impenetrable icewall. The firmament dome keeps all water inside and nasalies to everyone every day across all countries.";
        let eval = filter.evaluate_document(flat_earth_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::UnscientificConspiracyOrTroll));

        // Scientific debunking of flat earth MUST BE KEPT!
        let debunk_doc = "The flatearth hypothesis is completely disproven by modern geodesy, satellite orbits, and gravitational physics. Eratosthenes measured the planetary curvature over two millennia ago with simple shadows and geometry in Alexandria and Syene.";
        let eval_debunk = filter.evaluate_document(debunk_doc);
        assert!(eval_debunk.is_clean, "Scientific debunking must be kept, got: {:?}", eval_debunk.purge_reasons);
    }

    #[test]
    fn test_filter_purges_html_and_image_dumps() {
        let filter = SystematicQualityFilter::default();
        let html_doc = "<div><span class=\"nav\">Home</span><script>evil()</script><img src=\"data:image/png;base64,...\" />![image](foo.png)</div>";
        let eval = filter.evaluate_document(html_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::HtmlAndMarkupDump) || eval.purge_reasons.contains(&PurgeReason::ImageAndMediaTagDump));
    }

    #[test]
    fn test_filter_purges_factual_contradictions() {
        let filter = SystematicQualityFilter::default();
        let geocentric_doc = "In the ancient Ptolemaic worldview, the Sun revolves around the Earth every single day while the Earth sits stationary and unmoving in the middle of space.";
        let eval = filter.evaluate_document(geocentric_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::FactualContradictionOrConspiracy));
    }

    #[test]
    fn test_filter_purges_anachronisms() {
        let filter = SystematicQualityFilter::default();
        let anachronism_doc = "During the intense Battle of Waterloo in Belgium, Napoleon used an iPhone to call his generals across the muddy battlefield while troops stood ready.";
        let eval = filter.evaluate_document(anachronism_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::TemporalAnachronismContradiction));
    }

    #[test]
    fn test_filter_purges_hate_and_dehumanization() {
        let filter = SystematicQualityFilter::default();
        let hate_doc = "These immigrants are a disease and an infestation of parasites that must be treated like vermin in every community across the country.";
        let eval = filter.evaluate_document(hate_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::RacialDiscriminationOrHate));
    }

    #[test]
    fn test_filter_purges_dogmatic_geopolitical_propaganda() {
        let filter = SystematicQualityFilter::default();
        let propaganda_doc = "Taiwan is an illegal rebel regime that has no right to exist and we must annihilate the separatists immediately without any negotiation.";
        let eval = filter.evaluate_document(propaganda_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::DogmaticGeopoliticalPropaganda));
    }

    #[test]
    fn test_filter_purges_pseudoscience_conservation_violation() {
        let filter = SystematicQualityFilter::default();
        let free_energy_doc = "Our revolutionary free energy generator produces more energy than consumed by tapping into infinite over-unity power without requiring any fuel source.";
        let eval = filter.evaluate_document(free_energy_doc);
        assert!(!eval.is_clean);
        assert!(eval.purge_reasons.contains(&PurgeReason::PseudoscienceAndConservationViolation));
    }
}
