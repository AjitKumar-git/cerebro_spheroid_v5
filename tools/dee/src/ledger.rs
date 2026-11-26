use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::profiler::{inspect_csv, investigate_csv};
use crate::schema::{validate_csv, SchemaContract, ViolationKind};

/// Finite evidence assessment states defined in DEE specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceState {
    /// Unassessed initial state.
    Unassessed,
    /// Fully consistent with structural rules, expectations, and domain bounds.
    Consistent,
    /// Discrepancy observed (e.g. range edge, schema mismatch, unexpected column, duplicate).
    DiscrepancyDetected,
    /// Under active multi-variable investigation; not a corruption verdict.
    UnderInvestigation,
    /// Unusual but validated as genuine real-world rarity.
    RareValid,
    /// Genuine edge-case or emerging pattern of scientific/modeling value.
    PotentialDiscovery,
    /// Highly likely signal degradation or sensor/data-entry artifact.
    LikelyCorruption,
    /// Verified invariant breach (e.g. missing required key, non-nullable violation, type error).
    ConfirmedError,
    /// Insufficient evidence to adjudicate without further cross-variable context.
    Unresolved,
}

impl EvidenceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unassessed => "Unassessed",
            Self::Consistent => "Consistent",
            Self::DiscrepancyDetected => "DiscrepancyDetected",
            Self::UnderInvestigation => "UnderInvestigation",
            Self::RareValid => "RareValid",
            Self::PotentialDiscovery => "PotentialDiscovery",
            Self::LikelyCorruption => "LikelyCorruption",
            Self::ConfirmedError => "ConfirmedError",
            Self::Unresolved => "Unresolved",
        }
    }

    /// Allowed state transition validator to ensure evidence progression integrity.
    pub fn can_transition_to(&self, next: EvidenceState) -> bool {
        match self {
            Self::Unassessed => true,
            Self::Consistent => matches!(
                next,
                Self::DiscrepancyDetected | Self::UnderInvestigation | Self::ConfirmedError
            ),
            Self::DiscrepancyDetected => matches!(
                next,
                Self::UnderInvestigation | Self::ConfirmedError | Self::Unresolved | Self::Consistent
            ),
            Self::UnderInvestigation => matches!(
                next,
                Self::RareValid
                    | Self::PotentialDiscovery
                    | Self::LikelyCorruption
                    | Self::ConfirmedError
                    | Self::Unresolved
                    | Self::Consistent
            ),
            Self::RareValid | Self::PotentialDiscovery => {
                matches!(next, Self::UnderInvestigation | Self::ConfirmedError)
            }
            Self::LikelyCorruption => {
                matches!(next, Self::ConfirmedError | Self::UnderInvestigation | Self::RareValid)
            }
            Self::ConfirmedError => false, // Terminal error invariant
            Self::Unresolved => matches!(next, Self::UnderInvestigation | Self::Consistent | Self::ConfirmedError),
        }
    }
}

/// Identifies the discrete scope of an evidence entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityScope {
    Dataset,
    Column(String),
    Row(usize),
    Cell { row: usize, column: String },
}

impl std::fmt::Display for EntityScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "Dataset"),
            Self::Column(col) => write!(f, "Column({col})"),
            Self::Row(r) => write!(f, "Row({r})"),
            Self::Cell { row, column } => write!(f, "Cell(row={row}, col={column})"),
        }
    }
}

/// Individual immutable audit ledger entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEntry {
    pub id: String,
    pub entity: EntityScope,
    pub current_state: EvidenceState,
    pub previous_state: Option<EvidenceState>,
    pub summary: String,
    pub observed_fact: String,
    pub rule_reference: Option<String>,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerSummary {
    pub total_entries: usize,
    pub counts: HashMap<String, usize>,
    pub has_confirmed_errors: bool,
    pub under_investigation_count: usize,
    pub discrepancy_count: usize,
    pub consistent_count: usize,
}

/// Complete Evidence Ledger for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLedger {
    pub source: String,
    pub entries: Vec<EvidenceEntry>,
    pub summary: LedgerSummary,
    pub principles: Vec<String>,
}

impl EvidenceLedger {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
            entries: Vec::new(),
            summary: LedgerSummary {
                total_entries: 0,
                counts: HashMap::new(),
                has_confirmed_errors: false,
                under_investigation_count: 0,
                discrepancy_count: 0,
                consistent_count: 0,
            },
            principles: vec![
                "Marginal rarity is not a corruption verdict.".to_string(),
                "Observations remain under investigation until multi-variable context clarifies intent.".to_string(),
                "Source data is read-only; audits produce evidence records without mutating input.".to_string(),
                "Confirmed errors are reserved for hard invariant/schema contract violations.".to_string(),
                "Insufficient evidence terminates in Unresolved rather than forced classification.".to_string(),
            ],
        }
    }

    pub fn add_entry(
        &mut self,
        entity: EntityScope,
        state: EvidenceState,
        summary: &str,
        observed_fact: &str,
        rule_reference: Option<&str>,
        context: HashMap<String, String>,
    ) {
        let id = format!("EV-{:04}", self.entries.len() + 1);
        self.entries.push(EvidenceEntry {
            id,
            entity,
            current_state: state,
            previous_state: Some(EvidenceState::Unassessed),
            summary: summary.to_string(),
            observed_fact: observed_fact.to_string(),
            rule_reference: rule_reference.map(|s| s.to_string()),
            context,
        });
        self.recompute_summary();
    }

    pub fn recompute_summary(&mut self) {
        let mut counts = HashMap::new();
        let mut has_confirmed_errors = false;
        let mut under_investigation_count = 0;
        let mut discrepancy_count = 0;
        let mut consistent_count = 0;

        for entry in &self.entries {
            *counts.entry(entry.current_state.as_str().to_string()).or_insert(0) += 1;
            match entry.current_state {
                EvidenceState::ConfirmedError => has_confirmed_errors = true,
                EvidenceState::UnderInvestigation => under_investigation_count += 1,
                EvidenceState::DiscrepancyDetected => discrepancy_count += 1,
                EvidenceState::Consistent => consistent_count += 1,
                _ => {}
            }
        }

        self.summary = LedgerSummary {
            total_entries: self.entries.len(),
            counts,
            has_confirmed_errors,
            under_investigation_count,
            discrepancy_count,
            consistent_count,
        };
    }

    pub fn to_pretty_text(&self) -> String {
        let mut out = String::new();
        out.push_str("================================================================================\n");
        out.push_str(&format!("DEE EVIDENCE LEDGER (M1 Audit)\nSource: {}\n", self.source));
        out.push_str(&format!(
            "Total Entries: {} | Consistent: {} | Discrepancies: {} | Under Investigation: {} | Errors: {}\n",
            self.summary.total_entries,
            self.summary.consistent_count,
            self.summary.discrepancy_count,
            self.summary.under_investigation_count,
            if self.summary.has_confirmed_errors { "YES" } else { "None" }
        ));
        out.push_str("================================================================================\n\n");

        if self.entries.is_empty() {
            out.push_str("Ledger is empty. No evaluations recorded.\n");
            return out;
        }

        out.push_str("EVIDENCE LOG:\n");
        for entry in &self.entries {
            out.push_str(&format!(
                "[{}] {:<20} | Scope: {:<25} | {}\n",
                entry.id,
                entry.current_state.as_str(),
                entry.entity.to_string(),
                entry.summary
            ));
            out.push_str(&format!("       Fact: {}\n", entry.observed_fact));
            if let Some(r) = &entry.rule_reference {
                out.push_str(&format!("       Rule: {}\n", r));
            }
            if !entry.context.is_empty() {
                out.push_str("       Context: ");
                let pairs: Vec<String> = entry.context.iter().map(|(k, v)| format!("{k}={v}")).collect();
                out.push_str(&pairs.join(", "));
                out.push('\n');
            }
            out.push('\n');
        }

        out.push_str("DEE CORE PRINCIPLES:\n");
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

/// Generates a comprehensive Evidence Ledger for a CSV dataset, combining structural profiling,
/// schema contract validation (if schema provided), and MAD robust statistical rarity detection.
pub fn generate_evidence_ledger(
    csv_path: &Path,
    schema: Option<&SchemaContract>,
    rarity_threshold: f64,
) -> Result<EvidenceLedger, String> {
    let mut ledger = EvidenceLedger::new(&csv_path.display().to_string());

    // 1. Structural Profile Inspection
    let profile_report = inspect_csv(csv_path)?;

    // Check duplicate rows at dataset scope
    if profile_report.exact_duplicate_rows > 0 {
        let mut ctx = HashMap::new();
        ctx.insert("exact_duplicate_rows".to_string(), profile_report.exact_duplicate_rows.to_string());
        ctx.insert("total_rows".to_string(), profile_report.rows.to_string());
        ledger.add_entry(
            EntityScope::Dataset,
            EvidenceState::DiscrepancyDetected,
            "Exact duplicate rows detected in dataset",
            &format!("Found {} exact duplicate rows out of {} total rows", profile_report.exact_duplicate_rows, profile_report.rows),
            Some("Dataset Uniqueness Check"),
            ctx,
        );
    }

    // 2. Schema Contract Validation
    let mut columns_with_violations = std::collections::HashSet::new();
    if let Some(schema_contract) = schema {
        let validation_report = validate_csv(csv_path, schema_contract)?;

        for violation in &validation_report.violations {
            let col_name = violation.column.clone().unwrap_or_else(|| "unknown".to_string());
            let (state, entity) = match &violation.kind {
                ViolationKind::MissingRequiredColumn => (
                    EvidenceState::ConfirmedError,
                    EntityScope::Column(col_name.clone()),
                ),
                ViolationKind::UnexpectedColumn => (
                    EvidenceState::DiscrepancyDetected,
                    EntityScope::Column(col_name.clone()),
                ),
                ViolationKind::NullConstraintViolation => (
                    EvidenceState::ConfirmedError,
                    match violation.row_index {
                        Some(r) => EntityScope::Cell { row: r, column: col_name.clone() },
                        None => EntityScope::Column(col_name.clone()),
                    },
                ),
                ViolationKind::TypeMismatch => (
                    EvidenceState::ConfirmedError,
                    match violation.row_index {
                        Some(r) => EntityScope::Cell { row: r, column: col_name.clone() },
                        None => EntityScope::Column(col_name.clone()),
                    },
                ),
                ViolationKind::DuplicateKeyViolation => (
                    EvidenceState::DiscrepancyDetected,
                    match violation.row_index {
                        Some(r) => EntityScope::Cell { row: r, column: col_name.clone() },
                        None => EntityScope::Column(col_name.clone()),
                    },
                ),
                ViolationKind::RangeConstraintViolation => (
                    EvidenceState::UnderInvestigation,
                    match violation.row_index {
                        Some(r) => EntityScope::Cell { row: r, column: col_name.clone() },
                        None => EntityScope::Column(col_name.clone()),
                    },
                ),
                ViolationKind::DomainConstraintViolation => (
                    EvidenceState::DiscrepancyDetected,
                    match violation.row_index {
                        Some(r) => EntityScope::Cell { row: r, column: col_name.clone() },
                        None => EntityScope::Column(col_name.clone()),
                    },
                ),
                ViolationKind::RowCountViolation => (
                    EvidenceState::ConfirmedError,
                    EntityScope::Dataset,
                ),
            };

            columns_with_violations.insert(col_name);

            let mut ctx = HashMap::new();
            if let Some(r) = violation.row_index {
                ctx.insert("row_index".to_string(), r.to_string());
            }
            ctx.insert("value_observed".to_string(), violation.value_observed.clone());
            ctx.insert("expected".to_string(), violation.expected.clone());

            ledger.add_entry(
                entity,
                state,
                &format!("Schema contract violation: {:?}", violation.kind),
                &violation.explanation,
                Some(&format!("SchemaContract: {}", schema_contract.schema_name)),
                ctx,
            );
        }
    }

    // 3. Statistical Rarity Detection (UnderInvestigation - never corruption verdict)
    let rarity_candidates = investigate_csv(csv_path, rarity_threshold)?;
    let mut columns_with_rarities = std::collections::HashSet::new();

    for candidate in &rarity_candidates {
        columns_with_rarities.insert(candidate.column.clone());
        let mut ctx = HashMap::new();
        ctx.insert("robust_z".to_string(), format!("{:.4}", candidate.robust_z));
        ctx.insert("value".to_string(), candidate.value.to_string());

        ledger.add_entry(
            EntityScope::Cell {
                row: candidate.row_index,
                column: candidate.column.clone(),
            },
            EvidenceState::UnderInvestigation,
            "Marginally unusual numeric observation flagged for investigation",
            &format!(
                "Value {} in column '{}' at row {} has robust-z {:.4} (threshold: {}). Flagged as unusual evidence, NOT corruption.",
                candidate.value, candidate.column, candidate.row_index, candidate.robust_z, rarity_threshold
            ),
            Some("MAD Robust-Z Outlier Detection"),
            ctx,
        );
    }

    // 4. Mark clean columns as Consistent
    for profile in &profile_report.profiles {
        if !columns_with_violations.contains(&profile.name)
            && !columns_with_rarities.contains(&profile.name)
        {
            let mut ctx = HashMap::new();
            ctx.insert("inferred_kind".to_string(), profile.inferred_kind.clone());
            ctx.insert("non_empty".to_string(), profile.non_empty.to_string());
            ctx.insert("missing".to_string(), profile.missing.to_string());

            ledger.add_entry(
                EntityScope::Column(profile.name.clone()),
                EvidenceState::Consistent,
                "Column satisfies structural checks and has zero flagged discrepancies",
                &format!("Column '{}' (kind: {}) verified clean across {} records", profile.name, profile.inferred_kind, profile.non_empty),
                Some("Structural Consistency Baseline"),
                ctx,
            );
        }
    }

    Ok(ledger)
}
