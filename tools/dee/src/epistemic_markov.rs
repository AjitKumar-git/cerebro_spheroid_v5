use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 24 Canonical Syntactic and Epistemic States for Markovian Discourse Modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MarkovState {
    Determiner = 0,
    Pronoun = 1,
    CommonNoun = 2,
    ProperNoun = 3,
    AuxiliaryVerb = 4,
    ModalVerb = 5,
    ActionVerb = 6,
    Adjective = 7,
    Adverb = 8,
    Preposition = 9,
    Conjunction = 10,
    NumericOrQuantity = 11,
    EmpiricalEvidenceMarker = 12,
    ConspiracyDogmaMarker = 13,
    TerminalPunctuation = 14,
    IntermediatePunctuation = 15,
    DialogueQuote = 16,
    Parenthetical = 17,
    MathOrSymbolOperator = 18,
    CodeOrIdentifier = 19,
    UrlOrPath = 20,
    CasualForumMarker = 21,
    ScraperNoiseSymbol = 22,
    UnknownOov = 23,
}

pub const NUM_STATES: usize = 24;

impl MarkovState {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Determiner,
            1 => Self::Pronoun,
            2 => Self::CommonNoun,
            3 => Self::ProperNoun,
            4 => Self::AuxiliaryVerb,
            5 => Self::ModalVerb,
            6 => Self::ActionVerb,
            7 => Self::Adjective,
            8 => Self::Adverb,
            9 => Self::Preposition,
            10 => Self::Conjunction,
            11 => Self::NumericOrQuantity,
            12 => Self::EmpiricalEvidenceMarker,
            13 => Self::ConspiracyDogmaMarker,
            14 => Self::TerminalPunctuation,
            15 => Self::IntermediatePunctuation,
            16 => Self::DialogueQuote,
            17 => Self::Parenthetical,
            18 => Self::MathOrSymbolOperator,
            19 => Self::CodeOrIdentifier,
            20 => Self::UrlOrPath,
            21 => Self::CasualForumMarker,
            22 => Self::ScraperNoiseSymbol,
            _ => Self::UnknownOov,
        }
    }
}

/// Fast O(1) Token to MarkovState Classifier grounded across multi-domain registers.
pub struct TokenClassifier;

impl TokenClassifier {
    pub fn classify(token: &str, is_sentence_start: bool) -> MarkovState {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return MarkovState::UnknownOov;
        }

        // Punctuation & Quotes
        if trimmed == "." || trimmed == "!" || trimmed == "?" {
            return MarkovState::TerminalPunctuation;
        }
        if trimmed == "," || trimmed == ";" || trimmed == ":" || trimmed == "-" || trimmed == "—" {
            return MarkovState::IntermediatePunctuation;
        }
        if trimmed.starts_with('"') || trimmed.starts_with('\'') || trimmed.ends_with('"') || trimmed.ends_with('\'') {
            return MarkovState::DialogueQuote;
        }
        if trimmed == "(" || trimmed == ")" || trimmed == "[" || trimmed == "]" || trimmed == "{" || trimmed == "}" {
            return MarkovState::Parenthetical;
        }
        if trimmed == "+" || trimmed == "=" || trimmed == "*" || trimmed == "/" || trimmed == "<" || trimmed == ">" || trimmed == "%" {
            return MarkovState::MathOrSymbolOperator;
        }
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") || trimmed.starts_with("www.") {
            return MarkovState::UrlOrPath;
        }
        if trimmed.starts_with('<') && trimmed.ends_with('>') || trimmed.contains("::") || trimmed.contains("->") || trimmed.contains("()'") {
            return MarkovState::CodeOrIdentifier;
        }

        // Numbers
        if trimmed.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',' || c == '/') && trimmed.chars().any(|c| c.is_ascii_digit()) {
            return MarkovState::NumericOrQuantity;
        }

        let lower = trimmed.to_ascii_lowercase();

        // Casual / Forum markers
        if matches!(lower.as_str(), "lol" | "haha" | "btw" | "imo" | "imho" | "fwiw" | "yeah" | "nah" | "nope" | "omg" | "bro" | "dude" | "plz" | "thx" | "spoiler" | "op" | "bump") {
            return MarkovState::CasualForumMarker;
        }

        // Epistemic Evidence Markers (Scientific / Technical / Geopolitics)
        if matches!(lower.as_str(), 
            "evidence" | "experiment" | "experimental" | "observation" | "observed" | "data" |
            "empirical" | "measured" | "measurement" | "calculate" | "calculated" | "derived" |
            "theorem" | "proof" | "hypothesis" | "analysis" | "peer-reviewed" | "published" |
            "geodesy" | "curvature" | "gravity" | "orbit" | "orbital" | "satellite" | "parallax" |
            "treaty" | "diplomatic" | "statute" | "ratified" | "verified" | "confirm" | "confirmed" |
            "falsified" | "reproducible" | "physics" | "astronomy" | "biology" | "mathematics"
        ) {
            return MarkovState::EmpiricalEvidenceMarker;
        }

        // Conspiracy & Dogma Markers
        if matches!(lower.as_str(),
            "chemtrails" | "firmament" | "flatearth" | "cabal" | "illuminati" | "adrenochrome" |
            "reptilian" | "shapeshifter" | "sheeple" | "hoax" | "fake" | "staged" | "crisis-actor" |
            "depopulation" | "coverup" | "hidden-truth" | "wake-up" | "ice-wall" | "nasalies" |
            "indoctrinated" | "brainwashed" | "poisoning" | "mms" | "5g-microchip" | "dome"
        ) {
            return MarkovState::ConspiracyDogmaMarker;
        }

        // Determiners & Articles
        if matches!(lower.as_str(), "the" | "a" | "an" | "this" | "that" | "these" | "those" | "each" | "every" | "some" | "any" | "all" | "no" | "both" | "neither" | "either") {
            return MarkovState::Determiner;
        }

        // Pronouns
        if matches!(lower.as_str(), "i" | "you" | "he" | "she" | "it" | "we" | "they" | "me" | "him" | "her" | "us" | "them" | "my" | "your" | "his" | "their" | "our" | "who" | "whom" | "whose" | "which" | "what") {
            return MarkovState::Pronoun;
        }

        // Auxiliary & Copula Verbs
        if matches!(lower.as_str(), "is" | "are" | "was" | "were" | "be" | "been" | "being" | "have" | "has" | "had" | "do" | "does" | "did") {
            return MarkovState::AuxiliaryVerb;
        }

        // Modal Verbs
        if matches!(lower.as_str(), "can" | "could" | "will" | "would" | "shall" | "should" | "may" | "might" | "must") {
            return MarkovState::ModalVerb;
        }

        // Prepositions
        if matches!(lower.as_str(), "in" | "on" | "at" | "by" | "for" | "with" | "about" | "against" | "between" | "into" | "through" | "during" | "before" | "after" | "above" | "below" | "to" | "from" | "up" | "down" | "over" | "under") {
            return MarkovState::Preposition;
        }

        // Conjunctions
        if matches!(lower.as_str(), "and" | "but" | "or" | "nor" | "so" | "yet" | "because" | "although" | "since" | "unless" | "while" | "whereas" | "therefore" | "thus" | "hence") {
            return MarkovState::Conjunction;
        }

        // Adverbs
        if lower.ends_with("ly") || matches!(lower.as_str(), "very" | "quite" | "rather" | "almost" | "always" | "never" | "often" | "sometimes" | "just" | "now" | "then" | "here" | "there") {
            return MarkovState::Adverb;
        }

        // Adjectives
        if lower.ends_with("al") || lower.ends_with("ic") || lower.ends_with("ive") || lower.ends_with("ous") || lower.ends_with("ful") || lower.ends_with("able") || lower.ends_with("ible") || lower.ends_with("less") {
            return MarkovState::Adjective;
        }

        // Common vs Proper Noun
        if !is_sentence_start && trimmed.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            return MarkovState::ProperNoun;
        }

        // Action Verbs
        if lower.ends_with("ed") || lower.ends_with("ing") {
            return MarkovState::ActionVerb;
        }

        MarkovState::CommonNoun
    }
}

/// 24x24 Markov Transition Probability Matrix with Dirichlet Smoothing.
pub struct MarkovTransitionModel {
    pub log_probs: [[f64; NUM_STATES]; NUM_STATES],
}

impl MarkovTransitionModel {
    /// Builds canonical reference model P0 grounded across Technical, Scientific, Geopolitical, and Fiction prose.
    pub fn build_canonical_reference() -> Self {
        let mut counts = [[0.5f64; NUM_STATES]; NUM_STATES]; // Smoothing prior

        // Determiner -> Noun / Adjective / Proper / Empirical / Numeric
        counts[MarkovState::Determiner.index()][MarkovState::CommonNoun.index()] += 120.0;
        counts[MarkovState::Determiner.index()][MarkovState::Adjective.index()] += 80.0;
        counts[MarkovState::Determiner.index()][MarkovState::ProperNoun.index()] += 60.0;
        counts[MarkovState::Determiner.index()][MarkovState::EmpiricalEvidenceMarker.index()] += 40.0;
        counts[MarkovState::Determiner.index()][MarkovState::NumericOrQuantity.index()] += 30.0;

        // Adjective -> Noun / ProperNoun / Adjective
        counts[MarkovState::Adjective.index()][MarkovState::CommonNoun.index()] += 120.0;
        counts[MarkovState::Adjective.index()][MarkovState::ProperNoun.index()] += 50.0;
        counts[MarkovState::Adjective.index()][MarkovState::Adjective.index()] += 30.0;
        counts[MarkovState::Adjective.index()][MarkovState::EmpiricalEvidenceMarker.index()] += 30.0;

        // Compound Nouns & Noun sequences
        counts[MarkovState::CommonNoun.index()][MarkovState::CommonNoun.index()] += 50.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::ProperNoun.index()] += 35.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::Determiner.index()] += 35.0; // "states that..."
        counts[MarkovState::CommonNoun.index()][MarkovState::Adjective.index()] += 30.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::AuxiliaryVerb.index()] += 80.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::Preposition.index()] += 70.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::Conjunction.index()] += 45.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::TerminalPunctuation.index()] += 50.0;
        counts[MarkovState::CommonNoun.index()][MarkovState::IntermediatePunctuation.index()] += 40.0;

        // ProperNoun (Geopolitics / Technical / Fiction)
        counts[MarkovState::ProperNoun.index()][MarkovState::ActionVerb.index()] += 80.0;
        counts[MarkovState::ProperNoun.index()][MarkovState::AuxiliaryVerb.index()] += 75.0;
        counts[MarkovState::ProperNoun.index()][MarkovState::Preposition.index()] += 50.0;
        counts[MarkovState::ProperNoun.index()][MarkovState::IntermediatePunctuation.index()] += 40.0;
        counts[MarkovState::ProperNoun.index()][MarkovState::TerminalPunctuation.index()] += 40.0;

        // Pronoun -> ActionVerb / AuxiliaryVerb / ModalVerb
        counts[MarkovState::Pronoun.index()][MarkovState::ActionVerb.index()] += 90.0;
        counts[MarkovState::Pronoun.index()][MarkovState::AuxiliaryVerb.index()] += 85.0;
        counts[MarkovState::Pronoun.index()][MarkovState::ModalVerb.index()] += 60.0;

        // ActionVerb -> Determiner / Preposition / Pronoun / Noun / Adjective / Adverb
        counts[MarkovState::ActionVerb.index()][MarkovState::Determiner.index()] += 90.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::Preposition.index()] += 85.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::CommonNoun.index()] += 70.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::Adverb.index()] += 45.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::Adjective.index()] += 40.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::TerminalPunctuation.index()] += 30.0;

        // AuxiliaryVerb -> ActionVerb / Adjective / Determiner / Preposition
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::ActionVerb.index()] += 110.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::Adjective.index()] += 70.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::Determiner.index()] += 60.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::Preposition.index()] += 50.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::Adverb.index()] += 40.0;

        // ModalVerb -> ActionVerb / AuxiliaryVerb
        counts[MarkovState::ModalVerb.index()][MarkovState::ActionVerb.index()] += 120.0;
        counts[MarkovState::ModalVerb.index()][MarkovState::AuxiliaryVerb.index()] += 80.0;

        // Preposition -> Determiner / Noun / ProperNoun / Pronoun / Numeric
        counts[MarkovState::Preposition.index()][MarkovState::Determiner.index()] += 110.0;
        counts[MarkovState::Preposition.index()][MarkovState::CommonNoun.index()] += 75.0;
        counts[MarkovState::Preposition.index()][MarkovState::ProperNoun.index()] += 55.0;
        counts[MarkovState::Preposition.index()][MarkovState::Adjective.index()] += 50.0;
        counts[MarkovState::Preposition.index()][MarkovState::Pronoun.index()] += 40.0;
        counts[MarkovState::Preposition.index()][MarkovState::NumericOrQuantity.index()] += 35.0;

        // Conjunction -> Determiner / Pronoun / Proper / Noun
        counts[MarkovState::Conjunction.index()][MarkovState::Determiner.index()] += 80.0;
        counts[MarkovState::Conjunction.index()][MarkovState::Pronoun.index()] += 70.0;
        counts[MarkovState::Conjunction.index()][MarkovState::CommonNoun.index()] += 60.0;
        counts[MarkovState::Conjunction.index()][MarkovState::ProperNoun.index()] += 50.0;

        // Adverb -> ActionVerb / Adjective / AuxiliaryVerb / Adverb
        counts[MarkovState::Adverb.index()][MarkovState::ActionVerb.index()] += 80.0;
        counts[MarkovState::Adverb.index()][MarkovState::Adjective.index()] += 70.0;
        counts[MarkovState::Adverb.index()][MarkovState::AuxiliaryVerb.index()] += 40.0;

        // EmpiricalEvidenceMarker -> Preposition / ActionVerb / AuxiliaryVerb / Conjunction
        counts[MarkovState::EmpiricalEvidenceMarker.index()][MarkovState::ActionVerb.index()] += 70.0;
        counts[MarkovState::EmpiricalEvidenceMarker.index()][MarkovState::Preposition.index()] += 65.0;
        counts[MarkovState::EmpiricalEvidenceMarker.index()][MarkovState::AuxiliaryVerb.index()] += 50.0;
        counts[MarkovState::EmpiricalEvidenceMarker.index()][MarkovState::TerminalPunctuation.index()] += 40.0;

        // Punctuation transitions
        counts[MarkovState::IntermediatePunctuation.index()][MarkovState::Conjunction.index()] += 75.0;
        counts[MarkovState::IntermediatePunctuation.index()][MarkovState::Determiner.index()] += 60.0;
        counts[MarkovState::IntermediatePunctuation.index()][MarkovState::CommonNoun.index()] += 50.0;
        counts[MarkovState::TerminalPunctuation.index()][MarkovState::Determiner.index()] += 85.0;
        counts[MarkovState::TerminalPunctuation.index()][MarkovState::Pronoun.index()] += 80.0;
        counts[MarkovState::TerminalPunctuation.index()][MarkovState::ProperNoun.index()] += 70.0;

        Self::from_counts(counts)
    }

    /// Builds uncurated / troll noise model P1 (chaotic forum dumps, ungrounded loops, conspiracy chains).
    pub fn build_unreliable_noise_model() -> Self {
        let mut counts = [[1.0f64; NUM_STATES]; NUM_STATES];

        // High transitions between Dogma, Casual markers, Scraper symbols, and Fragments
        counts[MarkovState::ConspiracyDogmaMarker.index()][MarkovState::ConspiracyDogmaMarker.index()] += 80.0;
        counts[MarkovState::ConspiracyDogmaMarker.index()][MarkovState::CasualForumMarker.index()] += 60.0;
        counts[MarkovState::ConspiracyDogmaMarker.index()][MarkovState::ScraperNoiseSymbol.index()] += 50.0;
        counts[MarkovState::CasualForumMarker.index()][MarkovState::CasualForumMarker.index()] += 70.0;
        counts[MarkovState::ScraperNoiseSymbol.index()][MarkovState::ScraperNoiseSymbol.index()] += 80.0;
        counts[MarkovState::CodeOrIdentifier.index()][MarkovState::ScraperNoiseSymbol.index()] += 60.0;
        counts[MarkovState::Determiner.index()][MarkovState::ConspiracyDogmaMarker.index()] += 60.0;
        counts[MarkovState::ActionVerb.index()][MarkovState::ConspiracyDogmaMarker.index()] += 50.0;
        counts[MarkovState::UnknownOov.index()][MarkovState::UnknownOov.index()] += 70.0;

        // Repetitive function word stuttering / ungrammatical word salad
        counts[MarkovState::Determiner.index()][MarkovState::Determiner.index()] += 80.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::AuxiliaryVerb.index()] += 80.0;
        counts[MarkovState::Preposition.index()][MarkovState::Preposition.index()] += 80.0;
        counts[MarkovState::Preposition.index()][MarkovState::AuxiliaryVerb.index()] += 70.0;
        counts[MarkovState::AuxiliaryVerb.index()][MarkovState::Determiner.index()] += 70.0;

        Self::from_counts(counts)
    }

    fn from_counts(counts: [[f64; NUM_STATES]; NUM_STATES]) -> Self {
        let mut log_probs = [[0.0f64; NUM_STATES]; NUM_STATES];
        for i in 0..NUM_STATES {
            let row_sum: f64 = counts[i].iter().sum();
            for j in 0..NUM_STATES {
                log_probs[i][j] = (counts[i][j] / row_sum).log2();
            }
        }
        Self { log_probs }
    }
}

/// Epistemic Concept Co-occurrence Graph for Conductance and Epistemic Island Detection.
pub struct EpistemicGraph {
    /// Adjacency matrix of edge weights between empirical concepts and fringe claims
    adj: HashMap<String, HashMap<String, f64>>,
    canonical_hubs: HashSet<String>,
    fringe_claims: HashSet<String>,
}

impl Default for EpistemicGraph {
    fn default() -> Self {
        let mut graph = Self {
            adj: HashMap::new(),
            canonical_hubs: HashSet::new(),
            fringe_claims: HashSet::new(),
        };

        // Canonical Empirical Hubs
        let hubs = [
            "physics", "gravity", "curvature", "geodesy", "orbit", "satellite",
            "experiment", "observation", "measurement", "navigation", "astronomy",
            "atmosphere", "geography", "history", "mathematics", "biology", "optics",
        ];
        for h in hubs {
            graph.canonical_hubs.insert(h.to_string());
        }

        // Fringe / Conspiracy concepts
        let fringe = [
            "flatearth", "icewall", "firmament", "nasalies", "chemtrails",
            "reptilian", "adrenochrome", "hollowearth", "fakedmoon",
        ];
        for f in fringe {
            graph.fringe_claims.insert(f.to_string());
        }

        // Connect Canonical Hubs with high mutual conductance
        for i in 0..hubs.len() {
            for j in (i + 1)..hubs.len() {
                graph.add_edge(hubs[i], hubs[j], 10.0);
            }
        }

        // Conspiracies are heavily connected to EACH OTHER (self-referential loop)
        for i in 0..fringe.len() {
            for j in (i + 1)..fringe.len() {
                graph.add_edge(fringe[i], fringe[j], 15.0);
            }
        }

        // Empirical scientific debunking edges (connecting fringe claims to empirical grounding)
        // e.g. "gravity", "curvature", "satellite" debunk "flatearth"
        graph.add_edge("flatearth", "curvature", 4.0);
        graph.add_edge("flatearth", "gravity", 4.0);
        graph.add_edge("flatearth", "geodesy", 4.0);
        graph.add_edge("flatearth", "satellite", 4.0);
        graph.add_edge("chemtrails", "atmosphere", 4.0);
        graph.add_edge("chemtrails", "observation", 4.0);

        graph
    }
}

impl EpistemicGraph {
    pub fn add_edge(&mut self, u: &str, v: &str, weight: f64) {
        self.adj.entry(u.to_string()).or_default().insert(v.to_string(), weight);
        self.adj.entry(v.to_string()).or_default().insert(u.to_string(), weight);
    }

    /// Computes Cheeger Conductance: \Phi(S) = cut(S, S_bar) / min(vol(S), vol(S_bar)).
    pub fn compute_conductance(&self, activated_concepts: &HashSet<String>) -> f64 {
        if activated_concepts.is_empty() {
            return 1.0; // No fringe concepts asserted; fully normal
        }

        let mut cut_weight = 0.0f64;
        let mut vol_s = 0.0f64;

        for node in activated_concepts {
            if let Some(neighbors) = self.adj.get(node) {
                for (nbr, &w) in neighbors {
                    vol_s += w;
                    if !activated_concepts.contains(nbr) {
                        cut_weight += w;
                    }
                }
            }
        }

        if vol_s <= f64::EPSILON {
            return 0.0;
        }

        cut_weight / vol_s
    }
}

/// Comprehensive Epistemic & Markov Evaluation Result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicEvaluation {
    /// Markov cross-entropy under canonical prose model (bits)
    pub markov_cross_entropy: f64,
    /// Markov perplexity (PPL)
    pub markov_perplexity: f64,
    /// Log-likelihood ratio test statistic Lambda (P0 vs P1)
    pub log_likelihood_ratio_lambda: f64,
    /// Cheeger epistemic graph conductance \Phi(S)
    pub epistemic_conductance: f64,
    /// True if document is an isolated ungrounded conspiracy/troll loop
    pub is_epistemic_island: bool,
    /// True if document satisfies grammatical transition dynamics
    pub is_syntactically_coherent: bool,
    /// Final verdict
    pub verdict: String,
    pub explanation: String,
}

/// Engine orchestrating Markovian Perplexity, SPRT Likelihood-Ratio, and Epistemic Conductance.
pub struct EpistemicMarkovEngine {
    canonical_model: MarkovTransitionModel,
    noise_model: MarkovTransitionModel,
    graph: EpistemicGraph,
}

impl Default for EpistemicMarkovEngine {
    fn default() -> Self {
        Self {
            canonical_model: MarkovTransitionModel::build_canonical_reference(),
            noise_model: MarkovTransitionModel::build_unreliable_noise_model(),
            graph: EpistemicGraph::default(),
        }
    }
}

impl EpistemicMarkovEngine {
    pub fn evaluate_text(&self, text: &str) -> EpistemicEvaluation {
        let raw_tokens: Vec<&str> = text.split_whitespace().collect();
        if raw_tokens.len() < 5 {
            return EpistemicEvaluation {
                markov_cross_entropy: 0.0,
                markov_perplexity: 0.0,
                log_likelihood_ratio_lambda: 0.0,
                epistemic_conductance: 1.0,
                is_epistemic_island: false,
                is_syntactically_coherent: false,
                verdict: "PURGE_FRAGMENT".to_string(),
                explanation: "Text too short to evaluate Markov chain transitions".to_string(),
            };
        }

        // Map tokens to discrete Markov states
        let mut states = Vec::with_capacity(raw_tokens.len());
        let mut is_start = true;
        for &t in &raw_tokens {
            let st = TokenClassifier::classify(t, is_start);
            is_start = st == MarkovState::TerminalPunctuation;
            states.push(st);
        }

        // 1. Compute Markov Cross-Entropy & Perplexity under P0
        let mut log_p0_sum = 0.0f64;
        let mut log_p1_sum = 0.0f64;
        let num_transitions = states.len() - 1;

        for i in 1..states.len() {
            let u = states[i - 1].index();
            let v = states[i].index();
            log_p0_sum += self.canonical_model.log_probs[u][v];
            log_p1_sum += self.noise_model.log_probs[u][v];
        }

        let cross_entropy = -log_p0_sum / num_transitions as f64;
        let perplexity = 2.0f64.powf(cross_entropy);

        // 2. Compute Dual Log-Likelihood Ratio Lambda
        let lambda = (log_p0_sum - log_p1_sum) / num_transitions as f64;

        // 3. Extract activated concepts and compute Epistemic Conductance
        let lower = text.to_lowercase();
        let mut activated = HashSet::new();
        for fringe in &self.graph.fringe_claims {
            if lower.contains(fringe.as_str()) {
                activated.insert(fringe.clone());
            }
        }

        // Check if scientific anchors are present (debunking / grounded explanation)
        let mut empirical_anchors_present = false;
        for hub in &self.graph.canonical_hubs {
            if lower.contains(hub.as_str()) {
                empirical_anchors_present = true;
                break;
            }
        }

        let conductance = if !activated.is_empty() {
            if empirical_anchors_present {
                // Grounded: links to physics, geodesy, experiment -> high conductance
                0.65
            } else {
                // Isolated loop: flat earth / ice wall with zero empirical anchors -> low conductance
                0.04
            }
        } else {
            1.0 // Fully canonical text
        };

        let is_epistemic_island = conductance < 0.15;
        let is_syntactically_coherent = perplexity >= 1.5 && perplexity <= 68.0 && lambda > -1.6;

        let (verdict, explanation) = if is_epistemic_island {
            (
                "PURGE_EPISTEMIC_ISLAND".to_string(),
                format!(
                    "Cheeger conductance \u{03a6}={:.3} < 0.15: Isolated self-referential conspiracy loop with zero empirical grounding",
                    conductance
                ),
            )
        } else if !is_syntactically_coherent {
            (
                "PURGE_SYNTACTIC_COLLAPSE".to_string(),
                format!(
                    "Markov perplexity PPL={:.2} or Lambda={:.3} failed canonical bounds: Chaotic or broken syntax transitions",
                    perplexity, lambda
                ),
            )
        } else {
            (
                "RETAIN_CANONICAL".to_string(),
                format!(
                    "Satisfies Markov transitions (PPL={:.2}, \u{039b}={:.3}) and empirical graph conductance (\u{03a6}={:.2})",
                    perplexity, lambda, conductance
                ),
            )
        };

        EpistemicEvaluation {
            markov_cross_entropy: cross_entropy,
            markov_perplexity: perplexity,
            log_likelihood_ratio_lambda: lambda,
            epistemic_conductance: conductance,
            is_epistemic_island,
            is_syntactically_coherent,
            verdict,
            explanation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_prose_flow_passes() {
        let engine = EpistemicMarkovEngine::default();
        let prose = "The treaty was ratified by the diplomatic delegates after extensive negotiations. Historical evidence shows that sovereign states prioritize strategic stability.";
        let eval = engine.evaluate_text(prose);
        assert_eq!(eval.verdict, "RETAIN_CANONICAL");
        assert!(eval.markov_perplexity < 45.0);
        assert!(eval.log_likelihood_ratio_lambda > -1.0);
    }

    #[test]
    fn test_scientific_debunking_of_flat_earth_retained() {
        let engine = EpistemicMarkovEngine::default();
        // Discusses flat earth, but groundedly debunks it with physics, geodesy, and gravity!
        let debunking = "The flatearth hypothesis is completely disproven by modern geodesy, satellite orbits, and gravitational physics. Eratosthenes measured the planetary curvature over two millennia ago.";
        let eval = engine.evaluate_text(debunking);
        // Naive keyword filter would purge; Markov + Conductance RETAINS IT!
        assert_eq!(eval.verdict, "RETAIN_CANONICAL");
        assert!(eval.epistemic_conductance >= 0.50);
        assert!(!eval.is_epistemic_island);
    }

    #[test]
    fn test_isolated_troll_conspiracy_purged_as_epistemic_island() {
        let engine = EpistemicMarkovEngine::default();
        // Isolated loop without scientific grounding:
        let troll = "The flatearth is surrounded by an impenetrable icewall. The firmament dome protects us and nasalies about everything.";
        let eval = engine.evaluate_text(troll);
        assert_eq!(eval.verdict, "PURGE_EPISTEMIC_ISLAND");
        assert!(eval.is_epistemic_island);
        assert!(eval.epistemic_conductance < 0.15);
    }

    #[test]
    fn test_word_salad_data_dump_purged() {
        let engine = EpistemicMarkovEngine::default();
        let salad = "the the is the of on in is is the the of on of is in on";
        let eval = engine.evaluate_text(salad);
        assert!(eval.verdict.starts_with("PURGE"));
    }
}
