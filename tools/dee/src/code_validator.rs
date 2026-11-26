use serde::{Deserialize, Serialize};

/// Result of code syntax & structural integrity verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIntegrityReport {
    pub is_valid: bool,
    pub detected_language: Option<String>,
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub comment_ratio: f64,
    pub is_minified_bundle: bool,
    pub is_truncated: bool,
    pub syntax_defects: Vec<String>,
}

/// Deterministic code syntax, bracket balance, and minified boilerplate analyzer.
pub struct CodeValidator;

impl CodeValidator {
    /// Evaluates a code block or script for structural integrity, bracket balance, and quality.
    pub fn validate_code(code: &str) -> CodeIntegrityReport {
        let mut defects = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let total_lines = lines.len();

        if total_lines == 0 || code.trim().is_empty() {
            return CodeIntegrityReport {
                is_valid: true,
                detected_language: None,
                total_lines: 0,
                code_lines: 0,
                comment_lines: 0,
                comment_ratio: 0.0,
                is_minified_bundle: false,
                is_truncated: false,
                syntax_defects: Vec::new(),
            };
        }

        // 1. Minified bundle detection (e.g., web assets with extreme line lengths)
        let mut max_line_len = 0usize;
        let mut long_lines = 0usize;
        for line in &lines {
            let len = line.len();
            if len > max_line_len {
                max_line_len = len;
            }
            if len > 350 {
                long_lines += 1;
            }
        }

        let is_minified = max_line_len > 500 || (long_lines > 0 && total_lines <= 5);
        if is_minified {
            defects.push(format!("Detected minified code bundle (max line length {} chars)", max_line_len));
        }

        // 2. Language heuristic detection & comment line counting
        let mut comment_lines = 0usize;
        let mut code_lines = 0usize;
        let mut detected_lang: Option<&str> = None;

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with("//")
                || trimmed.starts_with("#")
                || trimmed.starts_with("/*")
                || trimmed.starts_with("*")
                || trimmed.starts_with("--")
            {
                comment_lines += 1;
            } else {
                code_lines += 1;
            }

            if detected_lang.is_none() {
                if trimmed.starts_with("def ") || trimmed.starts_with("import ") || trimmed.starts_with("class ") && trimmed.ends_with(':') {
                    detected_lang = Some("python");
                } else if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") || trimmed.starts_with("use std::") {
                    detected_lang = Some("rust");
                } else if trimmed.starts_with("function ") || trimmed.starts_with("const ") || trimmed.starts_with("let ") {
                    detected_lang = Some("javascript");
                } else if trimmed.starts_with("#include ") {
                    detected_lang = Some("c_cpp");
                } else if trimmed.starts_with("SELECT ") || trimmed.starts_with("CREATE TABLE") {
                    detected_lang = Some("sql");
                }
            }
        }

        let comment_ratio = if code_lines + comment_lines > 0 {
            comment_lines as f64 / (code_lines + comment_lines) as f64
        } else {
            0.0
        };

        // 3. Balanced delimiter check: (), [], {}
        let mut paren_stack = Vec::new();
        let mut in_string = false;
        let mut string_delim = ' ';
        let mut prev_char = ' ';

        for ch in code.chars() {
            if in_string {
                if ch == string_delim && prev_char != '\\' {
                    in_string = false;
                }
            } else {
                match ch {
                    '"' | '\'' => {
                        in_string = true;
                        string_delim = ch;
                    }
                    '(' | '[' | '{' => {
                        paren_stack.push(ch);
                    }
                    ')' => {
                        if paren_stack.pop() != Some('(') {
                            defects.push("Unmatched closing parenthesis ')'".to_string());
                            break;
                        }
                    }
                    ']' => {
                        if paren_stack.pop() != Some('[') {
                            defects.push("Unmatched closing bracket ']'".to_string());
                            break;
                        }
                    }
                    '}' => {
                        if paren_stack.pop() != Some('{') {
                            defects.push("Unmatched closing brace '}'".to_string());
                            break;
                        }
                    }
                    _ => {}
                }
            }
            prev_char = ch;
        }

        if in_string {
            defects.push("Unclosed string literal".to_string());
        }

        let is_truncated = !paren_stack.is_empty();
        if is_truncated {
            defects.push(format!("Truncated code structure ({} unclosed delimiters: {:?})", paren_stack.len(), paren_stack));
        }

        let is_valid = defects.is_empty();

        CodeIntegrityReport {
            is_valid,
            detected_language: detected_lang.map(|s| s.to_string()),
            total_lines,
            code_lines,
            comment_lines,
            comment_ratio,
            is_minified_bundle: is_minified,
            is_truncated,
            syntax_defects: defects,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_python_passes() {
        let py = "def add(a: int, b: int) -> int:\n    # Computes sum of integers\n    return a + b\n";
        let report = CodeValidator::validate_code(py);
        assert!(report.is_valid);
        assert_eq!(report.detected_language.as_deref(), Some("python"));
        assert!(report.comment_ratio > 0.20);
        assert!(!report.is_minified_bundle);
    }

    #[test]
    fn test_truncated_code_fails() {
        let truncated = "fn main() {\n    let x = [1, 2, 3;\n";
        let report = CodeValidator::validate_code(truncated);
        assert!(!report.is_valid);
        assert!(report.is_truncated);
    }

    #[test]
    fn test_minified_bundle_fails() {
        let minified = format!("const a=1;const b=2;function c(d,e){{return d+e;}}{}", "x".repeat(600));
        let report = CodeValidator::validate_code(&minified);
        assert!(!report.is_valid);
        assert!(report.is_minified_bundle);
    }
}
