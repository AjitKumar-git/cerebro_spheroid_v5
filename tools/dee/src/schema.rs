use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    Numeric,
    Text,
    Categorical,
    Boolean,
    Empty,
}

impl ColumnType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "numeric" | "float" | "int" | "integer" | "number" => ColumnType::Numeric,
            "bool" | "boolean" | "boolean-like" => ColumnType::Boolean,
            "categorical" | "category" => ColumnType::Categorical,
            "empty" => ColumnType::Empty,
            _ => ColumnType::Text,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericRange {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnContract {
    pub name: String,
    pub expected_type: ColumnType,
    #[serde(default = "default_true")]
    pub nullable: bool,
    #[serde(default)]
    pub unique: bool,
    #[serde(default)]
    pub range: Option<NumericRange>,
    #[serde(default)]
    pub allowed_values: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaContract {
    pub schema_name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_true")]
    pub allow_extra_columns: bool,
    #[serde(default)]
    pub min_rows: Option<usize>,
    #[serde(default)]
    pub max_rows: Option<usize>,
    pub columns: Vec<ColumnContract>,
}

fn default_version() -> String {
    "1.0".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    MissingRequiredColumn,
    UnexpectedColumn,
    NullConstraintViolation,
    TypeMismatch,
    RangeConstraintViolation,
    DomainConstraintViolation,
    DuplicateKeyViolation,
    RowCountViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractViolation {
    pub row_index: Option<usize>,
    pub column: Option<String>,
    pub kind: ViolationKind,
    pub value_observed: String,
    pub expected: String,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub source: String,
    pub schema_name: String,
    pub total_rows: usize,
    pub total_columns: usize,
    pub is_valid: bool,
    pub violations: Vec<ContractViolation>,
}

impl ValidationReport {
    pub fn to_pretty_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "DEE Structural Validation Report\nSource: {}\nSchema: {}\nRows: {} | Columns: {}\nStatus: {}\nTotal Violations: {}\n\n",
            self.source,
            self.schema_name,
            self.total_rows,
            self.total_columns,
            if self.is_valid { "VALID ✓" } else { "DISCREPANCIES DETECTED ✗" },
            self.violations.len()
        ));

        if self.violations.is_empty() {
            out.push_str("All schema contracts satisfied.\n");
        } else {
            out.push_str("Violations:\n");
            for (idx, v) in self.violations.iter().enumerate() {
                let loc = match (v.row_index, &v.column) {
                    (Some(r), Some(c)) => format!("row={r}, col={c}"),
                    (None, Some(c)) => format!("col={c}"),
                    (Some(r), None) => format!("row={r}"),
                    (None, None) => "dataset-level".to_string(),
                };
                out.push_str(&format!(
                    "  {:03}. [{:?}] at {}: observed='{}', expected='{}'\n       {}\n",
                    idx + 1,
                    v.kind,
                    loc,
                    v.value_observed,
                    v.expected,
                    v.explanation
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

impl ColumnContract {
    pub fn new(name: &str, expected_type: ColumnType) -> Self {
        Self {
            name: name.to_string(),
            expected_type,
            nullable: true,
            unique: false,
            range: None,
            allowed_values: None,
        }
    }

    pub fn with_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.range = Some(NumericRange { min, max });
        self
    }

    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = Some(values);
        self
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }
}

impl SchemaContract {
    pub fn new(name: &str) -> Self {
        Self {
            schema_name: name.to_string(),
            version: "1.0".to_string(),
            allow_extra_columns: true,
            min_rows: None,
            max_rows: None,
            columns: Vec::new(),
        }
    }

    pub fn add_column(&mut self, col: ColumnContract) {
        self.columns.push(col);
    }

    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Schema file read error: {e}"))?;
        serde_json::from_reader(file).map_err(|e| format!("Schema JSON parse error: {e}"))
    }

    pub fn to_json_file(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Schema file write error: {e}"))?;
        serde_json::to_writer_pretty(file, self).map_err(|e| format!("Schema JSON serialize error: {e}"))
    }
}

pub fn infer_schema_from_csv(path: &Path, schema_name: Option<&str>) -> Result<SchemaContract, String> {
    let report = crate::profiler::inspect_csv(path)?;
    let mut columns = Vec::new();

    for p in report.profiles {
        let expected_type = match p.inferred_kind.as_str() {
            "numeric" => ColumnType::Numeric,
            "boolean-like" => ColumnType::Boolean,
            "categorical/text" => {
                if p.non_empty > 0 && p.non_empty <= 50 {
                    ColumnType::Categorical
                } else {
                    ColumnType::Text
                }
            }
            _ => ColumnType::Text,
        };

        let range = p.numeric.as_ref().and_then(|n| {
            if n.min.is_some() || n.max.is_some() {
                Some(NumericRange { min: n.min, max: n.max })
            } else {
                None
            }
        });

        columns.push(ColumnContract {
            name: p.name,
            expected_type,
            nullable: p.missing > 0,
            unique: false,
            range,
            allowed_values: None,
        });
    }

    Ok(SchemaContract {
        schema_name: schema_name.unwrap_or("inferred_schema").to_string(),
        version: "1.0".to_string(),
        allow_extra_columns: true,
        min_rows: Some(1),
        max_rows: None,
        columns,
    })
}

pub fn validate_csv(path: &Path, schema: &SchemaContract) -> Result<ValidationReport, String> {
    let mut rdr = csv::Reader::from_path(path).map_err(|e| format!("CSV read error: {e}"))?;
    let headers = rdr.headers().map_err(|e| format!("CSV header error: {e}"))?.clone();
    let header_list: Vec<String> = headers.iter().map(|s| s.trim().to_string()).collect();
    let header_set: HashSet<String> = header_list.iter().cloned().collect();

    let mut violations = Vec::new();
    let schema_col_map: HashMap<String, &ColumnContract> = schema.columns.iter().map(|c| (c.name.clone(), c)).collect();

    // 1. Check missing required columns
    for col in &schema.columns {
        if !header_set.contains(&col.name) {
            violations.push(ContractViolation {
                row_index: None,
                column: Some(col.name.clone()),
                kind: ViolationKind::MissingRequiredColumn,
                value_observed: "<missing>".to_string(),
                expected: col.name.clone(),
                explanation: format!("Required column '{}' is absent from CSV header", col.name),
            });
        }
    }

    // 2. Check unexpected columns if not allowed
    if !schema.allow_extra_columns {
        for h in &header_list {
            if !schema_col_map.contains_key(h) {
                violations.push(ContractViolation {
                    row_index: None,
                    column: Some(h.clone()),
                    kind: ViolationKind::UnexpectedColumn,
                    value_observed: h.clone(),
                    expected: "<none>".to_string(),
                    explanation: format!("Unexpected column '{}' found with allow_extra_columns=false", h),
                });
            }
        }
    }

    // Map header indices to contracts
    let indexed_contracts: Vec<Option<&ColumnContract>> = header_list
        .iter()
        .map(|h| schema_col_map.get(h).copied())
        .collect();

    let mut row_count = 0usize;
    let mut unique_seen: HashMap<String, HashSet<String>> = HashMap::new();

    // Initialize unique trackers
    for col in &schema.columns {
        if col.unique {
            unique_seen.insert(col.name.clone(), HashSet::new());
        }
    }

    for record_res in rdr.records() {
        let record = record_res.map_err(|e| format!("CSV row read error: {e}"))?;
        row_count += 1;
        let line_num = row_count + 1; // 1-based, 1 is header

        for (col_idx, contract_opt) in indexed_contracts.iter().enumerate() {
            let contract = match contract_opt {
                Some(c) => *c,
                None => continue,
            };

            let val_str = record.get(col_idx).unwrap_or("").trim();

            // Nullability check
            if val_str.is_empty() {
                if !contract.nullable {
                    violations.push(ContractViolation {
                        row_index: Some(line_num),
                        column: Some(contract.name.clone()),
                        kind: ViolationKind::NullConstraintViolation,
                        value_observed: "<empty>".to_string(),
                        expected: "non-empty".to_string(),
                        explanation: format!("Column '{}' has nullable=false but row has empty value", contract.name),
                    });
                }
                continue;
            }

            // Uniqueness check
            if contract.unique {
                if let Some(set) = unique_seen.get_mut(&contract.name) {
                    if !set.insert(val_str.to_string()) {
                        violations.push(ContractViolation {
                            row_index: Some(line_num),
                            column: Some(contract.name.clone()),
                            kind: ViolationKind::DuplicateKeyViolation,
                            value_observed: val_str.to_string(),
                            expected: "unique value".to_string(),
                            explanation: format!("Duplicate value '{}' for unique column '{}'", val_str, contract.name),
                        });
                    }
                }
            }

            // Allowed categorical domain check
            if let Some(allowed) = &contract.allowed_values {
                if !allowed.iter().any(|a| a.eq_ignore_ascii_case(val_str)) {
                    violations.push(ContractViolation {
                        row_index: Some(line_num),
                        column: Some(contract.name.clone()),
                        kind: ViolationKind::DomainConstraintViolation,
                        value_observed: val_str.to_string(),
                        expected: format!("{:?}", allowed),
                        explanation: format!("Value '{}' is not in allowed domain for column '{}'", val_str, contract.name),
                    });
                }
            }

            // Type and range check
            match contract.expected_type {
                ColumnType::Numeric => {
                    match val_str.parse::<f64>() {
                        Ok(num) if num.is_finite() => {
                            if let Some(range) = &contract.range {
                                if let Some(min) = range.min {
                                    if num < min {
                                        violations.push(ContractViolation {
                                            row_index: Some(line_num),
                                            column: Some(contract.name.clone()),
                                            kind: ViolationKind::RangeConstraintViolation,
                                            value_observed: val_str.to_string(),
                                            expected: format!(">= {min}"),
                                            explanation: format!("Value {num} violates minimum bound {min}"),
                                        });
                                    }
                                }
                                if let Some(max) = range.max {
                                    if num > max {
                                        violations.push(ContractViolation {
                                            row_index: Some(line_num),
                                            column: Some(contract.name.clone()),
                                            kind: ViolationKind::RangeConstraintViolation,
                                            value_observed: val_str.to_string(),
                                            expected: format!("<= {max}"),
                                            explanation: format!("Value {num} violates maximum bound {max}"),
                                        });
                                    }
                                }
                            }
                        }
                        _ => {
                            violations.push(ContractViolation {
                                row_index: Some(line_num),
                                column: Some(contract.name.clone()),
                                kind: ViolationKind::TypeMismatch,
                                value_observed: val_str.to_string(),
                                expected: "numeric (finite f64)".to_string(),
                                explanation: format!("Failed to parse '{}' as numeric in column '{}'", val_str, contract.name),
                            });
                        }
                    }
                }
                ColumnType::Boolean => {
                    if val_str.parse::<bool>().is_err() && !["1", "0", "t", "f", "yes", "no"].contains(&val_str.to_lowercase().as_str()) {
                        violations.push(ContractViolation {
                            row_index: Some(line_num),
                            column: Some(contract.name.clone()),
                            kind: ViolationKind::TypeMismatch,
                            value_observed: val_str.to_string(),
                            expected: "boolean (true/false/1/0)".to_string(),
                            explanation: format!("Value '{}' is not valid boolean in column '{}'", val_str, contract.name),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Row count constraints
    if let Some(min_r) = schema.min_rows {
        if row_count < min_r {
            violations.push(ContractViolation {
                row_index: None,
                column: None,
                kind: ViolationKind::RowCountViolation,
                value_observed: row_count.to_string(),
                expected: format!(">= {min_r}"),
                explanation: format!("Row count {row_count} is less than required minimum {min_r}"),
            });
        }
    }
    if let Some(max_r) = schema.max_rows {
        if row_count > max_r {
            violations.push(ContractViolation {
                row_index: None,
                column: None,
                kind: ViolationKind::RowCountViolation,
                value_observed: row_count.to_string(),
                expected: format!("<= {max_r}"),
                explanation: format!("Row count {row_count} exceeds allowed maximum {max_r}"),
            });
        }
    }

    let is_valid = violations.is_empty();
    Ok(ValidationReport {
        source: path.display().to_string(),
        schema_name: schema.schema_name.clone(),
        total_rows: row_count,
        total_columns: header_list.len(),
        is_valid,
        violations,
    })
}
