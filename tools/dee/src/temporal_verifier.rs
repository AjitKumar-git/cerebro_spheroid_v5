use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// James Allen's 13 Interval Relations between two temporal intervals A and B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllenRelation {
    Before,    // A strictly ends before B begins (e_A < s_B)
    After,     // Inverse of Before (s_A > e_B)
    Meets,     // A ends exactly when B begins (e_A = s_B)
    MetBy,     // Inverse of Meets (s_A = e_B)
    Overlaps,  // A begins before B, ends inside B (s_A < s_B < e_A < e_B)
    OverlappedBy, // Inverse of Overlaps
    Starts,    // A and B begin together, A ends earlier (s_A = s_B, e_A < e_B)
    StartedBy, // Inverse of Starts
    During,    // A is strictly inside B (s_B < s_A, e_A < e_B)
    Contains,  // Inverse of During
    Finishes,  // A ends with B, but began later (s_B < s_A, e_A = e_B)
    FinishedBy, // Inverse of Finishes
    Equals,    // A and B occupy identical interval (s_A = s_B, e_A = e_B)
}

/// A temporal interval with integer bounds (years, negative for BCE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalInterval {
    pub start: i32,
    pub end: i32,
}

impl TemporalInterval {
    pub fn new(start: i32, end: i32) -> Self {
        assert!(start <= end, "start year must be <= end year");
        Self { start, end }
    }

    /// Computes Allen's interval relation between self (A) and other (B).
    pub fn relation_to(&self, other: &Self) -> AllenRelation {
        if self.end < other.start {
            AllenRelation::Before
        } else if self.start > other.end {
            AllenRelation::After
        } else if self.end == other.start {
            AllenRelation::Meets
        } else if self.start == other.end {
            AllenRelation::MetBy
        } else if self.start == other.start && self.end == other.end {
            AllenRelation::Equals
        } else if self.start == other.start && self.end < other.end {
            AllenRelation::Starts
        } else if self.start == other.start && self.end > other.end {
            AllenRelation::StartedBy
        } else if self.start > other.start && self.end == other.end {
            AllenRelation::Finishes
        } else if self.start < other.start && self.end == other.end {
            AllenRelation::FinishedBy
        } else if self.start > other.start && self.end < other.end {
            AllenRelation::During
        } else if self.start < other.start && self.end > other.end {
            AllenRelation::Contains
        } else if self.start < other.start && self.end < other.end {
            AllenRelation::Overlaps
        } else {
            AllenRelation::OverlappedBy
        }
    }
}

/// A detected temporal anachronism violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalViolation {
    pub subject: String,
    pub object: String,
    pub subject_interval: TemporalInterval,
    pub object_interval: TemporalInterval,
    pub computed_relation: AllenRelation,
    pub explanation: String,
}

/// Result of temporal interval consistency validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVerificationReport {
    pub is_temporally_consistent: bool,
    pub entities_matched: usize,
    pub violations: Vec<TemporalViolation>,
    pub summary: String,
}

/// Grounded Temporal Consistency & Anachronism Verifier using Allen's Interval Algebra.
pub struct TemporalVerifier {
    /// Grounded knowledge base of canonical existence intervals [start, end].
    pub entity_intervals: HashMap<String, TemporalInterval>,
}

impl Default for TemporalVerifier {
    fn default() -> Self {
        let mut v = Self {
            entity_intervals: HashMap::new(),
        };
        v.initialize_canonical_timeline();
        v
    }
}

impl TemporalVerifier {
    /// Initializes verified historical, technological, and event intervals.
    fn initialize_canonical_timeline(&mut self) {
        // Historical figures
        self.entity_intervals.insert("julius caesar".to_string(), TemporalInterval::new(-100, -44));
        self.entity_intervals.insert("alexander the great".to_string(), TemporalInterval::new(-356, -323));
        self.entity_intervals.insert("aristotle".to_string(), TemporalInterval::new(-384, -322));
        self.entity_intervals.insert("napoleon".to_string(), TemporalInterval::new(1769, 1821));
        self.entity_intervals.insert("napoleon bonaparte".to_string(), TemporalInterval::new(1769, 1821));
        self.entity_intervals.insert("george washington".to_string(), TemporalInterval::new(1732, 1799));
        self.entity_intervals.insert("abraham lincoln".to_string(), TemporalInterval::new(1809, 1865));
        self.entity_intervals.insert("isaac newton".to_string(), TemporalInterval::new(1643, 1727));
        self.entity_intervals.insert("galileo galilei".to_string(), TemporalInterval::new(1564, 1642));
        self.entity_intervals.insert("albert einstein".to_string(), TemporalInterval::new(1879, 1955));

        // Technologies & Inventions
        self.entity_intervals.insert("steam engine".to_string(), TemporalInterval::new(1712, 2100));
        self.entity_intervals.insert("locomotive".to_string(), TemporalInterval::new(1804, 2100));
        self.entity_intervals.insert("telegraph".to_string(), TemporalInterval::new(1837, 2100));
        self.entity_intervals.insert("telephone".to_string(), TemporalInterval::new(1876, 2100));
        self.entity_intervals.insert("radio".to_string(), TemporalInterval::new(1895, 2100));
        self.entity_intervals.insert("airplane".to_string(), TemporalInterval::new(1903, 2100));
        self.entity_intervals.insert("transistor".to_string(), TemporalInterval::new(1947, 2100));
        self.entity_intervals.insert("internet".to_string(), TemporalInterval::new(1983, 2100));
        self.entity_intervals.insert("world wide web".to_string(), TemporalInterval::new(1989, 2100));
        self.entity_intervals.insert("smartphone".to_string(), TemporalInterval::new(2000, 2100));
        self.entity_intervals.insert("iphone".to_string(), TemporalInterval::new(2007, 2100));

        // Historical Events
        self.entity_intervals.insert("peloponnesian war".to_string(), TemporalInterval::new(-431, -404));
        self.entity_intervals.insert("american revolution".to_string(), TemporalInterval::new(1775, 1783));
        self.entity_intervals.insert("french revolution".to_string(), TemporalInterval::new(1789, 1799));
        self.entity_intervals.insert("american civil war".to_string(), TemporalInterval::new(1861, 1865));
        self.entity_intervals.insert("world war i".to_string(), TemporalInterval::new(1914, 1918));
        self.entity_intervals.insert("world war 1".to_string(), TemporalInterval::new(1914, 1918));
        self.entity_intervals.insert("world war ii".to_string(), TemporalInterval::new(1939, 1945));
        self.entity_intervals.insert("world war 2".to_string(), TemporalInterval::new(1939, 1945));
        self.entity_intervals.insert("apollo 11".to_string(), TemporalInterval::new(1969, 1969));
    }

    /// Verifies temporal consistency across co-occurring entities in a text document.
    pub fn verify_text(&self, text: &str) -> TemporalVerificationReport {
        let lower = text.to_lowercase();
        let mut matched_entities = Vec::new();

        for (name, interval) in &self.entity_intervals {
            if lower.contains(name) {
                matched_entities.push((name.clone(), *interval));
            }
        }

        let mut violations = Vec::new();

        // Interaction verbs indicating contemporary co-existence or direct tool usage
        let usage_indicators = [
            "used", "using", "invented", "ordered by", "carried", "called on",
            "texted", "browsed", "tweeted", "drove", "flew", "deployed"
        ];
        let has_usage_claim = usage_indicators.iter().any(|v| lower.contains(v));

        if matched_entities.len() >= 2 && has_usage_claim {
            for i in 0..matched_entities.len() {
                for j in 0..matched_entities.len() {
                    if i == j {
                        continue;
                    }
                    let (ref name_a, int_a) = matched_entities[i];
                    let (ref name_b, int_b) = matched_entities[j];

                    let rel = int_a.relation_to(&int_b);

                    // If Person A died strictly BEFORE Technology/Event B began:
                    if rel == AllenRelation::Before {
                        // Check if text asserts direct contemporaneity or usage
                        // e.g., "Napoleon used an iPhone"
                        let span_a = lower.find(name_a);
                        let span_b = lower.find(name_b);

                        if let (Some(pos_a), Some(pos_b)) = (span_a, span_b) {
                            let dist = (pos_a as isize - pos_b as isize).abs();
                            if dist < 120 {
                                violations.push(TemporalViolation {
                                    subject: name_a.clone(),
                                    object: name_b.clone(),
                                    subject_interval: int_a,
                                    object_interval: int_b,
                                    computed_relation: rel,
                                    explanation: format!(
                                        "Anachronism contradiction: '{}' ([{}, {}]) is strictly BEFORE '{}' ([{}, {}]). Impossible contemporary interaction.",
                                        name_a, int_a.start, int_a.end, name_b, int_b.start, int_b.end
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        let is_consistent = violations.is_empty();
        let summary = if is_consistent {
            format!("Temporally consistent across {} matched historical entities", matched_entities.len())
        } else {
            format!("Detected {} chronological anachronism violations", violations.len())
        };

        TemporalVerificationReport {
            is_temporally_consistent: is_consistent,
            entities_matched: matched_entities.len(),
            violations,
            summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_chronology_passes() {
        let verifier = TemporalVerifier::default();
        let text = "Napoleon Bonaparte led the French army during the French Revolution until his exile in 1821.";
        let report = verifier.verify_text(text);
        assert!(report.is_temporally_consistent);
        assert_eq!(report.violations.len(), 0);
    }

    #[test]
    fn test_anachronism_detected() {
        let verifier = TemporalVerifier::default();
        let text = "During the Battle of Waterloo, Napoleon used an iPhone to coordinate troops with his marshals.";
        let report = verifier.verify_text(text);
        assert!(!report.is_temporally_consistent, "Anachronism must be detected");
        assert!(report.violations.iter().any(|v| v.subject == "napoleon" && v.object == "iphone"));
    }
}
