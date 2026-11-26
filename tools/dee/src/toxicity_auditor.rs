use serde::{Deserialize, Serialize};

/// Report detailing dehumanization projections and hate speech detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityAuditReport {
    pub is_clean: bool,
    pub dehumanization_score: f64,
    pub detected_demographics: Vec<String>,
    pub biological_pestilence_matches: Vec<String>,
    pub stereotypic_essentialism_matches: Vec<String>,
    pub explanation: String,
}

/// Deterministic Dehumanization and Hate Speech Auditor.
pub struct ToxicityAuditor;

impl ToxicityAuditor {
    /// Evaluates text for dehumanization vectors (projecting demographic entities onto pestilence/parasites)
    /// and universal essentialist prejudice.
    pub fn audit_text(text: &str) -> ToxicityAuditReport {
        let lower = text.to_lowercase();

        // 1. Target demographic classes
        let demographic_keywords = [
            "jewish", "jews", "muslims", "muslim", "blacks", "black people",
            "whites", "white people", "asians", "asian people", "hispanics", "latinos",
            "immigrants", "refugees", "migrants", "minorities", "gypsies", "roma", "indigenous"
        ];

        let mut matched_demographics = Vec::new();
        for &demo in &demographic_keywords {
            if lower.contains(demo) {
                matched_demographics.push(demo.to_string());
            }
        }

        if matched_demographics.is_empty() {
            return ToxicityAuditReport {
                is_clean: true,
                dehumanization_score: 0.0,
                detected_demographics: Vec::new(),
                biological_pestilence_matches: Vec::new(),
                stereotypic_essentialism_matches: Vec::new(),
                explanation: "No demographic target entities detected".to_string(),
            };
        }

        // 2. Biological pestilence, parasite & disease semantic vectors
        let pestilence_keywords = [
            "infestation", "infested", "vermin", "parasite", "parasites", "parasitic",
            "leech", "leeches", "cancer", "plague", "subhuman", "subhumans",
            "cockroaches", "rabid animals", "filthy pests", "exterminate", "cleansing"
        ];

        let mut pestilence_matches = Vec::new();
        for &pest in &pestilence_keywords {
            if lower.contains(pest) {
                pestilence_matches.push(pest.to_string());
            }
        }

        // 3. Check proximity between demographic entity and pestilence vectors (within window of 15 words)
        let words: Vec<&str> = lower.split_whitespace().collect();
        let mut dehumanization_pairs = 0usize;

        for (i, &word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());
            for demo in &matched_demographics {
                if clean_word == demo.as_str() || demo.contains(clean_word) && clean_word.len() > 3 {
                    // Check local window [i-12, i+12]
                    let start = i.saturating_sub(12);
                    let end = (i + 13).min(words.len());
                    for j in start..end {
                        let neighbor = words[j].trim_matches(|c: char| c.is_ascii_punctuation());
                        if pestilence_keywords.contains(&neighbor) {
                            dehumanization_pairs += 1;
                        }
                    }
                }
            }
        }

        // 4. Stereotypic universal essentialism checks
        let essentialist_patterns = [
            "are inherently inferior", "are genetically criminal", "are natural parasites",
            "naturally lack intelligence", "are a biological threat", "are culturally subhuman"
        ];

        let mut essentialism_matches = Vec::new();
        for pat in &essentialist_patterns {
            if lower.contains(pat) {
                essentialism_matches.push(pat.to_string());
            }
        }

        let dehumanization_score = (dehumanization_pairs as f64 * 0.40 + essentialism_matches.len() as f64 * 0.60).min(1.0);
        let is_clean = dehumanization_score < 0.35 && essentialism_matches.is_empty();

        let explanation = if !is_clean {
            format!(
                "Dehumanization / Hate detected (Score: {:.2}): Demographic targets {:?} associated with pestilence terms {:?} and essentialist claims {:?}",
                dehumanization_score, matched_demographics, pestilence_matches, essentialism_matches
            )
        } else {
            "Text is free of demographic dehumanization vectors and essentialist hate".to_string()
        };

        ToxicityAuditReport {
            is_clean,
            dehumanization_score,
            detected_demographics: matched_demographics,
            biological_pestilence_matches: pestilence_matches,
            stereotypic_essentialism_matches: essentialism_matches,
            explanation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_civil_rights_history_passes() {
        let text = "During the 1960s, African Americans and other minorities fought for equal civil rights, voting rights, and legal protections under the United States Constitution.";
        let rep = ToxicityAuditor::audit_text(text);
        assert!(rep.is_clean);
        assert_eq!(rep.dehumanization_score, 0.0);
    }

    #[test]
    fn test_dehumanizing_hate_speech_detected() {
        let text = "These immigrants are a disease and an infestation of parasites that must be treated like vermin.";
        let rep = ToxicityAuditor::audit_text(text);
        assert!(!rep.is_clean);
        assert!(rep.dehumanization_score >= 0.40);
        assert!(rep.biological_pestilence_matches.contains(&"parasites".to_string()));
    }
}
