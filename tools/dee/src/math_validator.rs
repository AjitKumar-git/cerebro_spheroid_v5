use serde::{Deserialize, Serialize};

/// Result of mathematical & formula integrity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathIntegrityReport {
    pub is_valid: bool,
    pub inline_math_blocks: usize,
    pub display_math_blocks: usize,
    pub syntax_errors_count: usize,
    pub ocr_garbage_ratio: f64,
    pub defect_explanations: Vec<String>,
}

/// Fast deterministic symbolic mathematical syntax and formula consistency validator.
pub struct MathValidator;

impl MathValidator {
    /// Validates LaTeX mathematical delimiter balance, environment scopes, and filters OCR math junk.
    pub fn validate_math_text(text: &str) -> MathIntegrityReport {
        let mut defects = Vec::new();
        let bytes = text.as_bytes();
        let len = bytes.len();

        let mut inline_count = 0usize;
        let mut display_count = 0usize;
        let mut in_single_dollar = false;
        let mut in_double_dollar = false;

        let mut i = 0;
        while i < len {
            // Escaped dollar: \$ -> skip
            if bytes[i] == b'\\' && i + 1 < len && bytes[i + 1] == b'$' {
                i += 2;
                continue;
            }

            // Double dollar: $$
            if bytes[i] == b'$' && i + 1 < len && bytes[i + 1] == b'$' {
                if in_single_dollar {
                    defects.push("Encountered '$$' inside unclosed inline '$' block".to_string());
                }
                if in_double_dollar {
                    in_double_dollar = false;
                } else {
                    in_double_dollar = true;
                    display_count += 1;
                }
                i += 2;
                continue;
            }

            // Single dollar: $
            if bytes[i] == b'$' {
                if !in_double_dollar {
                    if in_single_dollar {
                        in_single_dollar = false;
                    } else {
                        in_single_dollar = true;
                        inline_count += 1;
                    }
                }
                i += 1;
                continue;
            }

            // Environment begin / end check
            if bytes[i] == b'\\' && text[i..].starts_with("\\begin{") {
                if let Some(close_brace) = text[i..].find('}') {
                    let env_name = &text[i + 7..i + close_brace];
                    let end_tag = format!("\\end{{{}}}", env_name);
                    if !text[i + close_brace..].contains(&end_tag) {
                        defects.push(format!("Unclosed LaTeX environment '\\begin{{{}}}'", env_name));
                    }
                }
            }

            i += 1;
        }

        if in_single_dollar {
            defects.push("Unclosed inline LaTeX math block '$'".to_string());
        }
        if in_double_dollar {
            defects.push("Unclosed display LaTeX math block '$$'".to_string());
        }

        // Bracket balance check for curly braces
        let mut brace_depth = 0i32;
        for &b in bytes {
            if b == b'{' {
                brace_depth += 1;
            } else if b == b'}' {
                brace_depth -= 1;
                if brace_depth < 0 {
                    defects.push("Unexpected closing brace '}' without opening '{'".to_string());
                    break;
                }
            }
        }
        if brace_depth > 0 {
            defects.push(format!("{} unclosed curly braces '{{' in text", brace_depth));
        }

        // OCR Math Garbage Filter (detect isolated broken unicode math operator strings)
        let ocr_symbols = [
            '±', '÷', '˜', '¬', '‰', '·', '√', '∝', '∞', '∟', '∠', '∣', '∥', '∬', '∭', '∮',
        ];
        let total_chars = text.chars().count().max(1);
        let ocr_sym_count = text.chars().filter(|c| ocr_symbols.contains(c)).count();
        let ocr_ratio = ocr_sym_count as f64 / total_chars as f64;

        if ocr_ratio > 0.04 && total_chars > 50 {
            defects.push(format!("Excessive isolated OCR math symbol density ({:.1}%)", ocr_ratio * 100.0));
        }

        let is_valid = defects.is_empty();

        MathIntegrityReport {
            is_valid,
            inline_math_blocks: inline_count,
            display_math_blocks: display_count,
            syntax_errors_count: defects.len(),
            ocr_garbage_ratio: ocr_ratio,
            defect_explanations: defects,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_math_equations_pass() {
        let valid_text = "The relativistic energy equation is given by $E = mc^2$, where $c$ is the speed of light. In tensor form: $$\\mathcal{L} = \\frac{1}{2} \\partial_\\mu \\phi \\partial^\\mu \\phi - V(\\phi)$$";
        let report = MathValidator::validate_math_text(valid_text);
        assert!(report.is_valid, "Expected valid math to pass: {:?}", report.defect_explanations);
        assert_eq!(report.inline_math_blocks, 2);
        assert_eq!(report.display_math_blocks, 1);
    }

    #[test]
    fn test_unclosed_math_detected() {
        let broken = "Here is an unclosed formula $E = mc^2 and some text without closing dollar.";
        let report = MathValidator::validate_math_text(broken);
        assert!(!report.is_valid);
        assert!(report.defect_explanations.iter().any(|d| d.contains("Unclosed inline")));
    }
}
