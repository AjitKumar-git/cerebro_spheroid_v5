use serde::{Deserialize, Serialize};

/// Evaluation result for geopolitical neutrality and perspective attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeopoliticalReport {
    pub is_balanced_or_attributed: bool,
    pub disputed_regions_detected: Vec<String>,
    pub attribution_score: f64,
    pub is_dogmatic_propaganda: bool,
    pub detected_perspectives: Vec<String>,
    pub explanation: String,
}

/// Perspective attribution and geopolitical balance analyzer.
pub struct GeopoliticalBalancer;

impl GeopoliticalBalancer {
    /// Evaluates text mentioning contested geopolitical regions to distinguish
    /// balanced, attributed diplomatic discourse from un-attributed maximalist propaganda.
    pub fn evaluate_text(text: &str) -> GeopoliticalReport {
        let lower = text.to_lowercase();
        let mut disputed_regions = Vec::new();

        // 1. Detect contested sovereignty entities
        if lower.contains("taiwan") || lower.contains("taipei") || lower.contains("republic of china") {
            disputed_regions.push("Taiwan-Strait".to_string());
        }
        if lower.contains("crimea") || lower.contains("donbas") || lower.contains("donetsk") || lower.contains("luhansk") || lower.contains("ukraine") {
            disputed_regions.push("Ukraine-Crimea-Donbas".to_string());
        }
        if lower.contains("kashmir") || lower.contains("line of control") || lower.contains("ladakh") {
            disputed_regions.push("Kashmir-Himalayan".to_string());
        }
        if lower.contains("south china sea") || lower.contains("nine-dash line") || lower.contains("spratly islands") || lower.contains("paracel islands") {
            disputed_regions.push("South-China-Sea".to_string());
        }
        if lower.contains("falkland islands") || lower.contains("malvinas") {
            disputed_regions.push("Falklands-Malvinas".to_string());
        }
        if lower.contains("tibet") || lower.contains("dharamsala") {
            disputed_regions.push("Tibet".to_string());
        }

        if disputed_regions.is_empty() {
            return GeopoliticalReport {
                is_balanced_or_attributed: true,
                disputed_regions_detected: Vec::new(),
                attribution_score: 1.0,
                is_dogmatic_propaganda: false,
                detected_perspectives: Vec::new(),
                explanation: "No contested geopolitical entities detected".to_string(),
            };
        }

        // 2. Attribution markers (scholarly, diplomatic, international law framing)
        let attribution_markers = [
            "according to", "claims that", "maintains that", "officially recognizes",
            "under international law", "un resolution", "united nations", "general assembly",
            "resolution", "territorial integrity", "de facto", "de jure",
            "disputed by", "from the perspective of", "beijing states", "taipei maintains",
            "ukraine contends", "both parties dispute", "contested status", "sovereignty claim",
            "treaty of", "one china policy", "diplomatic relations", "unclos", "arbitration"
        ];

        let mut attribution_hits = 0usize;
        for marker in &attribution_markers {
            if lower.contains(marker) {
                attribution_hits += 1;
            }
        }

        // 3. Dogmatic, delegitimizing propaganda markers
        let propaganda_markers = [
            "has no right to exist", "must be liberated by force", "fake puppet state",
            "illegal rebel regime", "subservient lapdogs", "foreign lapdogs", "inseparable territory without negotiation",
            "crush all independence", "annihilate the separatists", "traitorous regime",
            "total territorial conquest", "must be crushed"
        ];

        let mut propaganda_hits = 0usize;
        for marker in &propaganda_markers {
            if lower.contains(marker) {
                propaganda_hits += 1;
            }
        }

        let total_indicators = (attribution_hits + propaganda_hits).max(1);
        let attribution_score = attribution_hits as f64 / total_indicators as f64;

        let is_dogmatic = propaganda_hits >= 1 && attribution_score < 0.35;
        let is_balanced = !is_dogmatic;

        let explanation = if is_dogmatic {
            format!(
                "Dogmatic nationalist propaganda detected in {:?} (Attribution Score: {:.2}, {} delegitimizing terms)",
                disputed_regions, attribution_score, propaganda_hits
            )
        } else {
            format!(
                "Balanced or diplomatically attributed geopolitical discourse across {:?} (Attribution Score: {:.2})",
                disputed_regions, attribution_score
            )
        };

        GeopoliticalReport {
            is_balanced_or_attributed: is_balanced,
            disputed_regions_detected: disputed_regions,
            attribution_score,
            is_dogmatic_propaganda: is_dogmatic,
            detected_perspectives: Vec::new(),
            explanation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attributed_geopolitics_passes() {
        let text = "Under the One China Policy, Beijing maintains that Taiwan is part of China, while Taipei operates as a self-governing democracy under international trade relations.";
        let rep = GeopoliticalBalancer::evaluate_text(text);
        assert!(rep.is_balanced_or_attributed, "Attributed diplomatic discourse must pass");
        assert!(!rep.is_dogmatic_propaganda);
        assert!(rep.attribution_score >= 0.70);
    }

    #[test]
    fn test_dogmatic_propaganda_purged() {
        let text = "Taiwan is an illegal rebel regime that has no right to exist and we must annihilate the separatists immediately.";
        let rep = GeopoliticalBalancer::evaluate_text(text);
        assert!(!rep.is_balanced_or_attributed, "Dogmatic propaganda must be flagged");
        assert!(rep.is_dogmatic_propaganda);
    }
}
