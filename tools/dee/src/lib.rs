pub mod code_validator;
pub mod epistemic_markov;
pub mod factual_verifier;
pub mod geopolitical_balancer;
pub mod ledger;
pub mod math_validator;
pub mod minhash_lsh;
pub mod parallel_pipeline;
pub mod pedagogical_scorer;
pub mod profiler;
pub mod pseudoscience_filter;
pub mod quality_filter;
pub mod schema;
pub mod temporal_verifier;
pub mod text_evaluator;
pub mod toxicity_auditor;

// Re-exports for convenience
pub use code_validator::{CodeIntegrityReport, CodeValidator};
pub use epistemic_markov::{
    EpistemicEvaluation, EpistemicGraph, EpistemicMarkovEngine, MarkovState,
    MarkovTransitionModel,
};
pub use factual_verifier::{
    Complex, FactualVerificationResult, FactualVerifier, FactualViolation, KnowledgeTriplet,
};
pub use geopolitical_balancer::{GeopoliticalBalancer, GeopoliticalReport};
pub use ledger::{
    generate_evidence_ledger, EntityScope, EvidenceEntry, EvidenceLedger, EvidenceState,
    LedgerSummary,
};
pub use math_validator::{MathIntegrityReport, MathValidator};
pub use minhash_lsh::{
    DeduplicationSummary, DisjointSet, MinHashLshConfig, MinHashLshEngine,
};
pub use parallel_pipeline::{ParallelPipeline, PipelineSummary};
pub use pedagogical_scorer::{
    CurriculumStage, PedagogicalScore, PedagogicalScorer,
};
pub use profiler::{
    inspect_csv, investigate_csv, ColumnProfile, DatasetReport, NumericSummary, RarityCandidate,
};
pub use pseudoscience_filter::{PseudoscienceFilter, PseudoscienceReport};
pub use quality_filter::{
    filter_dataset, DocumentQualityScore, FilterSummary, GrammarScorer, PurgeReason,
    SystematicQualityFilter,
};
pub use schema::{
    infer_schema_from_csv, validate_csv, ColumnContract, ColumnType, ContractViolation,
    NumericRange, SchemaContract, ValidationReport, ViolationKind,
};
pub use temporal_verifier::{
    AllenRelation, TemporalInterval, TemporalVerificationReport, TemporalVerifier,
    TemporalViolation,
};
pub use text_evaluator::{
    evaluate_text_column, TextAnomaly, TextMetrics, TextQualityReport,
};
pub use toxicity_auditor::{ToxicityAuditReport, ToxicityAuditor};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn inspect_and_detect_rarity_without_corruption_label() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id,value,group").unwrap();
        writeln!(f, "1,10,A").unwrap();
        writeln!(f, "2,11,A").unwrap();
        writeln!(f, "3,9,A").unwrap();
        writeln!(f, "4,10,A").unwrap();
        writeln!(f, "5,100,B").unwrap();
        let report = inspect_csv(&path).unwrap();
        assert_eq!(report.rows, 5);
        assert_eq!(report.columns, 3);
        assert_eq!(report.profiles[1].inferred_kind, "numeric");
        let candidates = investigate_csv(&path, 3.5).unwrap();
        assert!(candidates.iter().any(|c| c.value == 100.0));
        assert!(candidates
            .iter()
            .all(|c| !c.interpretation.to_lowercase().contains("corrupt verdict")));
    }

    #[test]
    fn test_schema_inference_and_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id,age,country,score").unwrap();
        writeln!(f, "1,25,USA,88.5").unwrap();
        writeln!(f, "2,30,CAN,92.0").unwrap();
        writeln!(f, "3,45,USA,75.0").unwrap();

        let schema = infer_schema_from_csv(&path, Some("user_schema")).unwrap();
        assert_eq!(schema.columns.len(), 4);

        // Validation against self should succeed
        let val = validate_csv(&path, &schema).unwrap();
        assert!(val.is_valid);
        assert_eq!(val.violations.len(), 0);
    }

    #[test]
    fn test_schema_contract_violations() {
        let dir = tempfile::tempdir().unwrap();
        let schema_path = dir.path().join("schema.json");
        let csv_path = dir.path().join("data.csv");

        let mut contract = SchemaContract::new("test_contract");
        contract.add_column(
            ColumnContract::new("token_count", ColumnType::Numeric)
                .with_range(Some(1.0), Some(100.0)),
        );
        contract.add_column(
            ColumnContract::new("stage", ColumnType::Categorical)
                .with_allowed_values(vec!["stage1".to_string(), "stage2".to_string()]),
        );
        contract.to_json_file(&schema_path).unwrap();

        // Write violating CSV
        let mut f = std::fs::File::create(&csv_path).unwrap();
        writeln!(f, "token_count,stage").unwrap();
        writeln!(f, "50,stage1").unwrap();
        writeln!(f, "250,stage3").unwrap(); // 250 > 100 max range, stage3 not allowed

        let loaded_contract = SchemaContract::from_json_file(&schema_path).unwrap();
        let report = validate_csv(&csv_path, &loaded_contract).unwrap();
        assert!(!report.is_valid);
        assert_eq!(report.violations.len(), 2);
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::RangeConstraintViolation)));
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::DomainConstraintViolation)));
    }

    #[test]
    fn test_evidence_ledger_generation() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("experiment.csv");

        let mut f = std::fs::File::create(&csv_path).unwrap();
        writeln!(f, "step,loss,metric").unwrap();
        writeln!(f, "1,2.5,10.0").unwrap();
        writeln!(f, "2,2.4,10.1").unwrap();
        writeln!(f, "3,2.3,9.9").unwrap();
        writeln!(f, "4,2.2,10.2").unwrap();
        writeln!(f, "5,2.1,10.0").unwrap();
        writeln!(f, "6,1.0,99.9").unwrap(); // metric 99.9 is an outlier

        let ledger = generate_evidence_ledger(&csv_path, None, 3.5).unwrap();
        assert_eq!(ledger.entries.len(), 3); // 2 consistent columns (step, loss), 1 outlier (metric) under investigation
        assert!(ledger.summary.under_investigation_count >= 1);
        assert!(!ledger.summary.has_confirmed_errors);

        // Pretty print text verification
        let text = ledger.to_pretty_text();
        assert!(text.contains("UnderInvestigation"));
        assert!(text.contains("Consistent"));
    }
}
