use serde::{Deserialize, Serialize};

/// Report evaluating pseudoscience, quackery, and physical conservation law compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoscienceReport {
    pub is_scientifically_grounded: bool,
    pub thermodynamic_violations: Vec<String>,
    pub avogadro_limit_violations: Vec<String>,
    pub quackery_cures_detected: Vec<String>,
    pub is_debunking_refutation: bool,
    pub explanation: String,
}

/// Deterministic Conservation Law & Pseudoscience Filter.
pub struct PseudoscienceFilter;

impl PseudoscienceFilter {
    /// Evaluates text for physical law violations (thermodynamics, Avogadro limit)
    /// and ungrounded medical quackery claims.
    pub fn evaluate_text(text: &str) -> PseudoscienceReport {
        let lower = text.to_lowercase();

        // 1. Check for genuine scientific debunking / regulatory consumer warning context
        let debunking_indicators = [
            "debunked", "disproven", "pseudoscience", "quackery", "fda warning",
            "ftc warned", "lacks clinical evidence", "placebo effect", "scientific consensus",
            "no active molecules", "violates the laws of physics", "fraudulent claim"
        ];
        let is_debunking = debunking_indicators.iter().any(|d| lower.contains(d));

        // 2. Thermodynamic Conservation Violations (Perpetual motion, free energy)
        let thermo_patterns = [
            "free energy generator", "perpetual motion machine", "over-unity device",
            "runs forever without fuel", "produces more energy than consumed",
            "zero point energy powers home without fuel", "infinite energy device"
        ];

        let mut thermo_violations = Vec::new();
        for &pat in &thermo_patterns {
            if lower.contains(pat) && !is_debunking {
                thermo_violations.push(format!("Thermodynamic Invariant Violation: Claim of '{}' asserts \u{0394}S < 0 or \u{0394}E \u{2260} 0", pat));
            }
        }

        // 3. Avogadro Dilution Limit Violations (Water memory, infinite dilutions)
        let avogadro_patterns = [
            "homeopathic 30c", "homeopathic 200c", "dilution beyond avogadro",
            "water memory cures", "water retains the memory of the molecule",
            "the more dilute the more potent"
        ];

        let mut avogadro_violations = Vec::new();
        for &pat in &avogadro_patterns {
            if lower.contains(pat) && !is_debunking {
                avogadro_violations.push(format!("Avogadro Limit Violation: Claim of '{}' exceeds N_A (10^-24), leaving statistically zero solute molecules", pat));
            }
        }

        // 4. Medical Quackery & Fraudulent Cures
        let quackery_patterns = [
            "miracle mineral supplement cures", "alkaline water cures cancer",
            "quantum frequency heals tumors", "detox foot pads extract all toxins",
            "neutralizes 5g cellular radiation from blood", "cancels chemotherapy with baking soda"
        ];

        let mut quackery_cures = Vec::new();
        for &pat in &quackery_patterns {
            if lower.contains(pat) && !is_debunking {
                quackery_cures.push(format!("Unregulated Medical Quackery: Dangerous fraudulent health assertion '{}'", pat));
            }
        }

        let is_grounded = thermo_violations.is_empty()
            && avogadro_violations.is_empty()
            && quackery_cures.is_empty();

        let explanation = if is_debunking {
            "Text contains legitimate scientific debunking or regulatory consumer protection context".to_string()
        } else if !is_grounded {
            let mut summary_parts = Vec::new();
            if !thermo_violations.is_empty() {
                summary_parts.push(format!("{} thermodynamic violations", thermo_violations.len()));
            }
            if !avogadro_violations.is_empty() {
                summary_parts.push(format!("{} Avogadro dilution violations", avogadro_violations.len()));
            }
            if !quackery_cures.is_empty() {
                summary_parts.push(format!("{} medical quackery assertions", quackery_cures.len()));
            }
            format!("Pseudoscience detected: {}", summary_parts.join(", "))
        } else {
            "Text adheres to physical conservation laws and scientific grounding".to_string()
        };

        PseudoscienceReport {
            is_scientifically_grounded: is_grounded,
            thermodynamic_violations: thermo_violations,
            avogadro_limit_violations: avogadro_violations,
            quackery_cures_detected: quackery_cures,
            is_debunking_refutation: is_debunking,
            explanation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermodynamic_violation_flagged() {
        let text = "Our new free energy generator produces more energy than consumed by tapping into infinite over-unity power.";
        let rep = PseudoscienceFilter::evaluate_text(text);
        assert!(!rep.is_scientifically_grounded);
        assert!(!rep.thermodynamic_violations.is_empty());
    }

    #[test]
    fn test_homeopathy_avogadro_violation_flagged() {
        let text = "This homeopathic 30c remedy works because water retains the memory of the molecule even after complete dilution.";
        let rep = PseudoscienceFilter::evaluate_text(text);
        assert!(!rep.is_scientifically_grounded);
        assert!(!rep.avogadro_limit_violations.is_empty());
    }

    #[test]
    fn test_medical_debunking_retained() {
        let text = "The FDA warned that claims asserting a miracle mineral supplement cures diseases are dangerous pseudoscience lacking clinical evidence.";
        let rep = PseudoscienceFilter::evaluate_text(text);
        assert!(rep.is_scientifically_grounded);
        assert!(rep.is_debunking_refutation);
    }
}
