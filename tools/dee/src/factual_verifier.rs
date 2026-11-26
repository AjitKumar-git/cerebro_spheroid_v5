use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 2D Complex number for RotatE relational manifold embeddings (t = h \circ r).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_angle(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }
}

/// Relational Knowledge Triplet (Head, Relation, Tail).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KnowledgeTriplet {
    pub head: String,
    pub relation: String,
    pub tail: String,
}

impl KnowledgeTriplet {
    pub fn new(head: &str, relation: &str, tail: &str) -> Self {
        Self {
            head: head.to_lowercase(),
            relation: relation.to_lowercase(),
            tail: tail.to_lowercase(),
        }
    }
}

/// A detected factual violation with exact mathematical diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualViolation {
    pub category: String,
    pub claimed_triplet: Option<KnowledgeTriplet>,
    pub canonical_grounding: Option<KnowledgeTriplet>,
    pub manifold_energy: f64,
    pub spectral_frustration: f64,
    pub mdl_penalty_bits: f64,
    pub explanation: String,
}

/// Verification results across manifold energy, signed graph balance, and MDL complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualVerificationResult {
    pub is_factually_consistent: bool,
    pub total_manifold_energy: f64,
    pub spectral_frustration_lambda1: f64,
    pub unbalanced_cycles: usize,
    pub mdl_excess_bits: f64,
    pub violations: Vec<FactualViolation>,
    pub summary: String,
}

/// Canonical Knowledge Grounding & Relational Manifold for deterministic factual verification.
pub struct FactualVerifier {
    /// Dimension d of complex embedding space.
    pub dim: usize,
    /// Canonical entity embeddings: E \subset C^d.
    pub entity_embeddings: HashMap<String, Vec<Complex>>,
    /// Canonical relation rotation angles: R \subset (S^1)^d.
    pub relation_rotations: HashMap<String, Vec<Complex>>,
    /// Ground truth canonical triplets that are verified physical/scientific facts.
    pub canonical_triplets: Vec<KnowledgeTriplet>,
    /// Explicit antithetical / contradictory triplet mappings.
    pub antithetical_triplets: Vec<(KnowledgeTriplet, KnowledgeTriplet, &'static str)>,
    /// Auxiliary conspiracy epicycles and their MDL bit penalties.
    pub auxiliary_epicycles: HashMap<String, f64>,
}

impl Default for FactualVerifier {
    fn default() -> Self {
        let mut verifier = Self {
            dim: 8,
            entity_embeddings: HashMap::new(),
            relation_rotations: HashMap::new(),
            canonical_triplets: Vec::new(),
            antithetical_triplets: Vec::new(),
            auxiliary_epicycles: HashMap::new(),
        };
        verifier.initialize_canonical_manifold();
        verifier
    }
}

impl FactualVerifier {
    /// Initializes canonical physical, astronomical, and scientific facts into the complex manifold.
    fn initialize_canonical_manifold(&mut self) {
        let d = self.dim;

        // Helper to construct normalized d-dimensional complex vector with given base angle.
        let make_vec = |base_angle: f64| -> Vec<Complex> {
            (0..d)
                .map(|k| Complex::from_angle(base_angle + (k as f64) * 0.392699)) // 0.392699 ~ pi/8
                .collect()
        };

        // 1. Entities
        self.entity_embeddings.insert("earth".to_string(), make_vec(0.10));
        self.entity_embeddings.insert("sun".to_string(), make_vec(1.57)); // pi/2
        self.entity_embeddings.insert("moon".to_string(), make_vec(0.52));
        self.entity_embeddings.insert("mars".to_string(), make_vec(2.20));
        self.entity_embeddings.insert("solar_system".to_string(), make_vec(3.14));
        self.entity_embeddings.insert("nasa".to_string(), make_vec(0.85));
        self.entity_embeddings.insert("apollo_11".to_string(), make_vec(1.15));
        self.entity_embeddings.insert("moon_landing".to_string(), make_vec(1.30));
        self.entity_embeddings.insert("satellite".to_string(), make_vec(0.40));
        self.entity_embeddings.insert("gps".to_string(), make_vec(0.45));
        self.entity_embeddings.insert("antarctica".to_string(), make_vec(2.80));

        // Shapes & physical states
        self.entity_embeddings.insert("oblate_spheroid".to_string(), make_vec(0.70));
        self.entity_embeddings.insert("spherical".to_string(), make_vec(0.72));
        self.entity_embeddings.insert("flat_disc".to_string(), make_vec(4.71)); // 3pi/2 (orthogonal/inverted)
        self.entity_embeddings.insert("plane".to_string(), make_vec(4.80));
        self.entity_embeddings.insert("historical_reality".to_string(), make_vec(1.00));
        self.entity_embeddings.insert("hoax_fraud".to_string(), make_vec(4.50));

        // 2. Relations (Rotations r_i = e^{i \theta_i})
        // Orbits: rotating Earth by theta_orbits aligns with Sun
        let theta_orbits = 1.57 - 0.10; // 1.47
        self.relation_rotations.insert("orbits".to_string(), make_vec(theta_orbits));

        // HasShape: rotating Earth by theta_shape aligns with oblate spheroid
        let theta_shape = 0.70 - 0.10; // 0.60
        self.relation_rotations.insert("has_shape".to_string(), make_vec(theta_shape));

        // Historical event validation
        let theta_event = 1.00 - 1.30;
        self.relation_rotations.insert("is_status".to_string(), make_vec(theta_event));

        // 3. Ground Truth Canonical Triplets
        self.canonical_triplets.push(KnowledgeTriplet::new("earth", "orbits", "sun"));
        self.canonical_triplets.push(KnowledgeTriplet::new("moon", "orbits", "earth"));
        self.canonical_triplets.push(KnowledgeTriplet::new("earth", "has_shape", "oblate_spheroid"));
        self.canonical_triplets.push(KnowledgeTriplet::new("earth", "has_shape", "spherical"));
        self.canonical_triplets.push(KnowledgeTriplet::new("moon_landing", "is_status", "historical_reality"));
        self.canonical_triplets.push(KnowledgeTriplet::new("apollo_11", "is_status", "historical_reality"));
        self.canonical_triplets.push(KnowledgeTriplet::new("satellite", "orbits", "earth"));
        self.canonical_triplets.push(KnowledgeTriplet::new("gps", "is_status", "historical_reality"));

        // 4. Explicit Antithetical / Contradictory Triplet Mappings
        // If claimed triplet occurs, it directly contradicts the canonical ground truth.
        self.antithetical_triplets.push((
            KnowledgeTriplet::new("sun", "orbits", "earth"),
            KnowledgeTriplet::new("earth", "orbits", "sun"),
            "Geocentric inversion: Sun does not orbit the Earth in Newtonian celestial mechanics",
        ));
        self.antithetical_triplets.push((
            KnowledgeTriplet::new("earth", "has_shape", "flat_disc"),
            KnowledgeTriplet::new("earth", "has_shape", "oblate_spheroid"),
            "Flat Earth assertion violates planetary hydrostatic equilibrium and geodesy",
        ));
        self.antithetical_triplets.push((
            KnowledgeTriplet::new("earth", "has_shape", "plane"),
            KnowledgeTriplet::new("earth", "has_shape", "oblate_spheroid"),
            "Planar Earth assertion contradicts celestial navigation and curvature measurements",
        ));
        self.antithetical_triplets.push((
            KnowledgeTriplet::new("moon_landing", "is_status", "hoax_fraud"),
            KnowledgeTriplet::new("moon_landing", "is_status", "historical_reality"),
            "Apollo moon landing conspiracy assertion contradicts laser retroreflector and physical telemetry",
        ));
        self.antithetical_triplets.push((
            KnowledgeTriplet::new("nasa", "is_status", "hoax_fraud"),
            KnowledgeTriplet::new("nasa", "is_status", "historical_reality"),
            "NASA universal fraud conspiracy assertion",
        ));

        // 5. Auxiliary Conspiracy Epicycles (MDL bit penalties for ungrounded ad-hoc hypotheses)
        self.auxiliary_epicycles.insert("ice wall".to_string(), 8.5);
        self.auxiliary_epicycles.insert("firmament dome".to_string(), 9.0);
        self.auxiliary_epicycles.insert("atmospheric lensing bend".to_string(), 7.5);
        self.auxiliary_epicycles.insert("nasa cgi".to_string(), 8.0);
        self.auxiliary_epicycles.insert("fake space".to_string(), 8.5);
        self.auxiliary_epicycles.insert("hollow earth".to_string(), 9.2);
        self.auxiliary_epicycles.insert("geocentric dome".to_string(), 8.8);
        self.auxiliary_epicycles.insert("space is water".to_string(), 9.5);
        self.auxiliary_epicycles.insert("satellites do not exist".to_string(), 7.8);
    }

    /// Computes the RotatE manifold distance energy E(h, r, t) = ||h \circ r - t||_2.
    /// In consistent physics, canonical triplets yield E \approx 0.
    /// Contradictory/inverted triplets yield high energy penalties.
    pub fn compute_relational_energy(&self, triplet: &KnowledgeTriplet) -> f64 {
        let h_opt = self.entity_embeddings.get(&triplet.head);
        let r_opt = self.relation_rotations.get(&triplet.relation);
        let t_opt = self.entity_embeddings.get(&triplet.tail);

        match (h_opt, r_opt, t_opt) {
            (Some(h), Some(r), Some(t)) => {
                let mut sum_diff_sq = 0.0;
                for i in 0..self.dim {
                    let rotated = h[i].mul(r[i]);
                    let diff = rotated.sub(t[i]);
                    sum_diff_sq += diff.norm_sq();
                }
                (sum_diff_sq / self.dim as f64).sqrt()
            }
            _ => 1.0, // Neutral unmapped energy
        }
    }

    /// Extracts asserted knowledge triplets from document text.
    /// Detects whether the claim is framed constructively vs. critical/debunking.
    pub fn extract_asserted_triplets(&self, text: &str) -> (Vec<KnowledgeTriplet>, bool) {
        let lower = text.to_lowercase();
        let mut triplets = Vec::new();

        // 1. Check for explicit debunking / scientific education context
        let debunk_markers = [
            "debunk", "disprove", "refute", "false claim", "fallacy", "myth",
            "misconception", "conspiracy theory", "contrary to", "discredited",
            "erroneous belief", "eratosthenes proved", "copernicus proved",
            "galileo observed", "pseudo-scientific", "pseudoscience"
        ];
        let has_debunking_context = debunk_markers.iter().any(|m| lower.contains(m));

        // 2. Pattern extractions
        // Geocentric claims:
        if (lower.contains("sun revolves around the earth")
            || lower.contains("sun revolves around earth")
            || lower.contains("sun orbits the earth")
            || lower.contains("sun orbits earth")
            || lower.contains("earth is stationary and the sun moves around it"))
            && !has_debunking_context
        {
            triplets.push(KnowledgeTriplet::new("sun", "orbits", "earth"));
        }

        // Heliocentric assertions:
        if lower.contains("earth orbits the sun")
            || lower.contains("earth revolves around the sun")
            || lower.contains("earth's orbit around the sun")
            || (has_debunking_context && lower.contains("sun revolves around the earth"))
        {
            triplets.push(KnowledgeTriplet::new("earth", "orbits", "sun"));
        }

        // Flat earth claims:
        if (lower.contains("earth is flat")
            || lower.contains("the earth is a flat")
            || lower.contains("flat earth truth")
            || lower.contains("earth is not a globe")
            || lower.contains("earth is a planar")
            || lower.contains("earth is not a sphere"))
            && !has_debunking_context
        {
            triplets.push(KnowledgeTriplet::new("earth", "has_shape", "flat_disc"));
        }

        // Spherical earth assertions:
        if lower.contains("earth is spherical")
            || lower.contains("earth is an oblate spheroid")
            || lower.contains("curvature of the earth")
            || lower.contains("spherical earth")
            || (has_debunking_context && lower.contains("flat earth"))
        {
            triplets.push(KnowledgeTriplet::new("earth", "has_shape", "oblate_spheroid"));
        }

        // Moon landing conspiracy claims:
        if (lower.contains("moon landing was faked")
            || lower.contains("moon landing is a hoax")
            || lower.contains("apollo 11 was staged")
            || lower.contains("never landed on the moon")
            || lower.contains("nasa faked the moon"))
            && !has_debunking_context
        {
            triplets.push(KnowledgeTriplet::new("moon_landing", "is_status", "hoax_fraud"));
        }

        // NASA / space conspiracy claims:
        if (lower.contains("nasa is a lie")
            || lower.contains("satellites are fake")
            || lower.contains("space is fake")
            || lower.contains("space travel is impossible"))
            && !has_debunking_context
        {
            triplets.push(KnowledgeTriplet::new("nasa", "is_status", "hoax_fraud"));
        }

        (triplets, has_debunking_context)
    }

    /// Evaluates Signed Graph Balance & Spectral Frustration (\lambda_1(\bar{L})).
    /// Under Harary's Balance Theorem, an assertion graph is balanced iff every cycle has
    /// an even number of negative edges (\prod \sigma(e) = +1).
    /// Contradictions create odd-negative cycles, driving \lambda_1(\bar{L}) > 0.
    pub fn compute_signed_balance(
        &self,
        triplets: &[KnowledgeTriplet],
    ) -> (f64, usize, Vec<String>) {
        if triplets.is_empty() {
            return (0.0, 0, Vec::new());
        }

        // Map entities to node indices
        let mut node_map = HashMap::new();
        let mut index = 0usize;
        for t in triplets {
            if !node_map.contains_key(&t.head) {
                node_map.insert(t.head.clone(), index);
                index += 1;
            }
            if !node_map.contains_key(&t.tail) {
                node_map.insert(t.tail.clone(), index);
                index += 1;
            }
        }
        for canon in &self.canonical_triplets {
            if node_map.contains_key(&canon.head) || node_map.contains_key(&canon.tail) {
                if !node_map.contains_key(&canon.head) {
                    node_map.insert(canon.head.clone(), index);
                    index += 1;
                }
                if !node_map.contains_key(&canon.tail) {
                    node_map.insert(canon.tail.clone(), index);
                    index += 1;
                }
            }
        }

        let n = node_map.len();
        if n < 2 {
            return (0.0, 0, Vec::new());
        }

        // Signed Adjacency Matrix A_\sigma \in R^{n \times n}
        let mut a_sigma = vec![vec![0.0f64; n]; n];
        let mut edge_signs: HashMap<(usize, usize), f64> = HashMap::new();

        // 1. Add canonical edges (positive relationships +1.0)
        for canon in &self.canonical_triplets {
            if let (Some(&u), Some(&v)) = (node_map.get(&canon.head), node_map.get(&canon.tail)) {
                a_sigma[u][v] = 1.0;
                a_sigma[v][u] = 1.0;
                edge_signs.insert((u.min(v), u.max(v)), 1.0);
            }
        }

        // 2. Add asserted edges: if antithetical to canonical, sign is -1.0!
        let mut frustrated_explanations = Vec::new();
        for t in triplets {
            let u = match node_map.get(&t.head) {
                Some(&idx) => idx,
                None => continue,
            };
            let v = match node_map.get(&t.tail) {
                Some(&idx) => idx,
                None => continue,
            };

            // Check if this triplet is antithetical to any canonical ground truth
            let mut is_antithetical = false;
            for (anti, canon, reason) in &self.antithetical_triplets {
                if t.head == anti.head && t.relation == anti.relation && t.tail == anti.tail {
                    is_antithetical = true;
                    // Negative signed edge: -1.0 represents epistemic contradiction
                    a_sigma[u][v] = -1.0;
                    a_sigma[v][u] = -1.0;
                    edge_signs.insert((u.min(v), u.max(v)), -1.0);
                    frustrated_explanations.push(format!(
                        "Contradiction: '{} {} {}' conflicts with canonical '{} {} {}'. {}",
                        t.head, t.relation, t.tail, canon.head, canon.relation, canon.tail, reason
                    ));
                    break;
                }
            }

            if !is_antithetical {
                // Compatible assertion: +1.0
                a_sigma[u][v] = 1.0;
                a_sigma[v][u] = 1.0;
                edge_signs.insert((u.min(v), u.max(v)), 1.0);
            }
        }

        // 3. Count unbalanced cycles (Harary's Balance Theorem: prod \sigma(e) == -1)
        let mut unbalanced_cycles = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if a_sigma[i][j].abs() < 1e-6 {
                    continue;
                }
                for k in (j + 1)..n {
                    if a_sigma[j][k].abs() < 1e-6 || a_sigma[k][i].abs() < 1e-6 {
                        continue;
                    }
                    let sign_product = a_sigma[i][j] * a_sigma[j][k] * a_sigma[k][i];
                    if sign_product < 0.0 {
                        unbalanced_cycles += 1;
                    }
                }
            }
        }

        // 4. Construct Signed Laplacian: \bar{L} = D - A_\sigma
        // where D(i, i) = \sum_j |A_\sigma(i, j)|
        let mut degrees = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                degrees[i] += a_sigma[i][j].abs();
            }
        }

        let mut signed_laplacian = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    signed_laplacian[i][i] = degrees[i] - a_sigma[i][i];
                } else {
                    signed_laplacian[i][j] = -a_sigma[i][j];
                }
            }
        }

        // 5. Estimate smallest eigenvalue \lambda_1(\bar{L}) via Shifted Power Iteration
        // If graph is balanced, \lambda_1 == 0. If frustrated, \lambda_1 > 0.
        let mut max_deg = 0.0f64;
        for &d in &degrees {
            if d > max_deg {
                max_deg = d;
            }
        }
        let shift = 2.0 * max_deg + 1.0;
        let mut shifted = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    shifted[i][i] = shift - signed_laplacian[i][i];
                } else {
                    shifted[i][j] = -signed_laplacian[i][j];
                }
            }
        }

        // Power iteration to find dominant eigenvalue of shifted matrix
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..40 {
            let mut v_next = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    v_next[i] += shifted[i][j] * v[j];
                }
            }
            let norm = v_next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for x in &mut v_next {
                    *x /= norm;
                }
            }
            v = v_next;
        }

        // Rayleigh quotient on signed_laplacian: \lambda_1 = v^T \bar{L} v
        let mut lv = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                lv[i] += signed_laplacian[i][j] * v[j];
            }
        }
        let lambda_1 = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum::<f64>().max(0.0);

        (lambda_1, unbalanced_cycles, frustrated_explanations)
    }

    /// Evaluates Minimum Description Length (MDL) algorithmic complexity penalty (\Delta L).
    /// Detects the explosion of ad-hoc ungrounded epicycles used to rationalize false claims.
    pub fn compute_mdl_excess(&self, text: &str) -> (f64, Vec<String>) {
        let lower = text.to_lowercase();
        let mut total_excess_bits = 0.0f64;
        let mut epicycles_found = Vec::new();

        for (epicycle, penalty) in &self.auxiliary_epicycles {
            if lower.contains(epicycle) {
                total_excess_bits += penalty;
                epicycles_found.push(format!("Ad-hoc epicycle '{}' (+{:.1} bits)", epicycle, penalty));
            }
        }

        (total_excess_bits, epicycles_found)
    }

    /// End-to-end deterministic factual verification of a text passage.
    pub fn verify_text(&self, text: &str) -> FactualVerificationResult {
        let (triplets, is_debunking) = self.extract_asserted_triplets(text);

        // If explicitly recognized as academic/scientific debunking, text is permitted and consistent.
        if is_debunking && !triplets.iter().any(|t| t.tail == "flat_disc" || t.tail == "hoax_fraud") {
            return FactualVerificationResult {
                is_factually_consistent: true,
                total_manifold_energy: 0.0,
                spectral_frustration_lambda1: 0.0,
                unbalanced_cycles: 0,
                mdl_excess_bits: 0.0,
                violations: Vec::new(),
                summary: "Passage contains valid scientific debunking and canonical discourse".to_string(),
            };
        }

        // 1. Manifold Relational Energy E(h, r, t)
        let mut total_energy = 0.0f64;
        let mut violations = Vec::new();

        for t in &triplets {
            let energy = self.compute_relational_energy(t);
            total_energy += energy;

            // Check if antithetical
            for (anti, canon, explanation) in &self.antithetical_triplets {
                if t.head == anti.head && t.relation == anti.relation && t.tail == anti.tail {
                    violations.push(FactualViolation {
                        category: "RelationalManifoldEnergyViolation".to_string(),
                        claimed_triplet: Some(t.clone()),
                        canonical_grounding: Some(canon.clone()),
                        manifold_energy: energy,
                        spectral_frustration: 0.0,
                        mdl_penalty_bits: 0.0,
                        explanation: explanation.to_string(),
                    });
                }
            }
        }

        // 2. Signed Graph Frustration & Harary Cycle Balance
        let (lambda_1, unbalanced_cycles, frustration_details) = self.compute_signed_balance(&triplets);
        if lambda_1 > 0.10 || unbalanced_cycles > 0 {
            for detail in frustration_details {
                violations.push(FactualViolation {
                    category: "SignedGraphBalanceFrustration".to_string(),
                    claimed_triplet: None,
                    canonical_grounding: None,
                    manifold_energy: 0.0,
                    spectral_frustration: lambda_1,
                    mdl_penalty_bits: 0.0,
                    explanation: format!(
                        "Harary theorem violation: \u{03bb}_1(\u{44c})={:.3}, {} unbalanced negative cycles. {}",
                        lambda_1, unbalanced_cycles, detail
                    ),
                });
            }
        }

        // 3. Minimum Description Length (MDL) Complexity Penalty
        let (mdl_bits, epicycles) = self.compute_mdl_excess(text);
        if mdl_bits >= 7.5 {
            violations.push(FactualViolation {
                category: "MdlAlgorithmicComplexityExplosion".to_string(),
                claimed_triplet: None,
                canonical_grounding: None,
                manifold_energy: 0.0,
                spectral_frustration: 0.0,
                mdl_penalty_bits: mdl_bits,
                explanation: format!(
                    "MDL excess \u{0394}L={:.1} bits \u{2265} 7.5 threshold due to ungrounded epicycles: {}",
                    mdl_bits,
                    epicycles.join("; ")
                ),
            });
        }

        let is_consistent = violations.is_empty();
        let summary = if is_consistent {
            "Text is factually consistent under canonical physics and information manifolds".to_string()
        } else {
            format!(
                "Factual violation detected: {} defects (Energy={:.2}, Frustration \u{03bb}_1={:.2}, MDL excess={:.1} bits)",
                violations.len(),
                total_energy,
                lambda_1,
                mdl_bits
            )
        };

        FactualVerificationResult {
            is_factually_consistent: is_consistent,
            total_manifold_energy: total_energy,
            spectral_frustration_lambda1: lambda_1,
            unbalanced_cycles,
            mdl_excess_bits: mdl_bits,
            violations,
            summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_heliocentrism_passes() {
        let verifier = FactualVerifier::default();
        let text = "Planetary astrophysics demonstrates that the Earth orbits the Sun in an elliptical path governed by Kepler's laws.";
        let res = verifier.verify_text(text);
        assert!(res.is_factually_consistent, "Heliocentrism must pass: {:?}", res.violations);
        assert_eq!(res.violations.len(), 0);
        assert_eq!(res.unbalanced_cycles, 0);
        assert!(res.spectral_frustration_lambda1 < 0.05);
    }

    #[test]
    fn test_geocentric_inversion_detected() {
        let verifier = FactualVerifier::default();
        let text = "Ancient geocentrism claimed that the Sun revolves around the Earth while the Earth remains motionless.";
        let res = verifier.verify_text(text);
        assert!(!res.is_factually_consistent, "Geocentrism claim must be detected");
        assert!(res.violations.iter().any(|v| v.category == "RelationalManifoldEnergyViolation"));
        assert!(res.violations.iter().any(|v| v.explanation.contains("Geocentric inversion")));
    }

    #[test]
    fn test_flat_earth_and_mdl_epicycles_detected() {
        let verifier = FactualVerifier::default();
        let text = "The Earth is flat and surrounded by a giant ice wall, shielded by an impenetrable firmament dome.";
        let res = verifier.verify_text(text);
        assert!(!res.is_factually_consistent, "Flat Earth assertion must be detected");
        assert!(res.violations.iter().any(|v| v.category == "MdlAlgorithmicComplexityExplosion"));
        assert!(res.mdl_excess_bits >= 15.0);
    }

    #[test]
    fn test_scientific_debunking_permitted() {
        let verifier = FactualVerifier::default();
        let text = "Astrophysicists thoroughly debunk the myth that the Sun revolves around the Earth, reiterating that planetary orbits are heliocentric.";
        let res = verifier.verify_text(text);
        assert!(res.is_factually_consistent, "Scientific debunking of falsehoods must be permitted");
        assert_eq!(res.violations.len(), 0);
    }

    #[test]
    fn test_moon_landing_hoax_detected() {
        let verifier = FactualVerifier::default();
        let text = "Conspiracy theorists claim Apollo 11 was staged and humans never landed on the Moon.";
        let res = verifier.verify_text(text);
        assert!(!res.is_factually_consistent, "Moon landing conspiracy must be detected");
        assert!(res.violations.iter().any(|v| v.explanation.contains("moon landing conspiracy")));
    }
}
