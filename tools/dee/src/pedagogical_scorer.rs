use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// SmolLM2 Pretraining Curriculum Stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CurriculumStage {
    /// Stage 1 (0%-55%): Fundamental, narrative, foundational concepts (FKGL 4.0 - 8.5)
    Stage1Fundamental,
    /// Stage 2 (55%-73%): Intermediate explanatory, high school STEM & humanities (FKGL 8.5 - 12.0)
    Stage2Intermediate,
    /// Stage 3 (73%-91%): Advanced analytical, undergraduate textbooks, technical guides (FKGL 12.0 - 16.0)
    Stage3Advanced,
    /// Stage 4 (91%-100%): Frontier research papers, formal proofs, academic specifications (FKGL > 16.0)
    Stage4Research,
}

impl CurriculumStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stage1Fundamental => "stage1_fundamental",
            Self::Stage2Intermediate => "stage2_intermediate",
            Self::Stage3Advanced => "stage3_advanced",
            Self::Stage4Research => "stage4_research",
        }
    }
}

/// Detailed pedagogical readability & curriculum evaluation report for a text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedagogicalScore {
    pub fkgl_grade_level: f64,
    pub flesch_reading_ease: f64,
    pub dale_chall_score: f64,
    pub academic_word_density: f64,
    pub words_count: usize,
    pub sentences_count: usize,
    pub syllables_count: usize,
    pub complex_words_ratio: f64,
    pub recommended_stage: CurriculumStage,
    pub pedagogical_summary: String,
}

/// Classical Dale-Chall 3000 common words root sample for readability scoring.
const DALE_CHALL_COMMON_LEMMAS: &[&str] = &[
    "a", "about", "above", "across", "act", "add", "afraid", "after", "again", "against", "age",
    "ago", "agree", "ahead", "air", "all", "allow", "almost", "alone", "along", "already", "also",
    "always", "am", "among", "an", "and", "animal", "another", "answer", "any", "appear", "are",
    "area", "arm", "around", "art", "as", "ask", "at", "away", "baby", "back", "bad", "ball",
    "bank", "base", "be", "bear", "beat", "beautiful", "became", "because", "become", "bed", "been",
    "before", "began", "begin", "behind", "believe", "bell", "belong", "below", "beside", "best",
    "better", "between", "big", "bird", "bit", "black", "block", "blood", "blow", "blue", "board",
    "boat", "body", "bone", "book", "born", "both", "bottom", "box", "boy", "branch", "bread",
    "break", "bright", "bring", "broad", "broke", "brother", "brought", "brown", "build", "burn",
    "busy", "but", "buy", "by", "call", "came", "camp", "can", "cannot", "car", "care", "carry",
    "case", "cat", "catch", "cause", "center", "century", "certain", "chair", "chance", "change",
    "charge", "check", "chief", "child", "children", "choose", "church", "circle", "city", "class",
    "clean", "clear", "climb", "clock", "close", "cloth", "cloud", "cold", "color", "come", "common",
    "company", "complete", "condition", "contain", "continue", "control", "cook", "cool", "copy",
    "corner", "correct", "cost", "could", "country", "course", "court", "cover", "cow", "crop",
    "cross", "crowd", "cry", "cup", "cut", "dance", "dark", "day", "dead", "deal", "dear", "death",
    "decide", "deep", "degree", "depend", "describe", "desert", "design", "desk", "determine",
    "develop", "die", "differ", "dinner", "direct", "discover", "distance", "divide", "do", "doctor",
    "dog", "dollar", "done", "door", "double", "doubt", "down", "draw", "dream", "dress", "drink",
    "drive", "drop", "dry", "during", "each", "ear", "early", "earth", "east", "easy", "eat", "edge",
    "effect", "effort", "egg", "eight", "either", "electric", "element", "else", "end", "enemy",
    "energy", "engine", "enough", "enter", "equal", "escape", "even", "evening", "event", "ever",
    "every", "exact", "example", "except", "expect", "experience", "explain", "eye", "face", "fact",
    "fair", "fall", "family", "famous", "far", "farm", "fast", "fat", "father", "fear", "feel",
    "feet", "fell", "felt", "few", "field", "fight", "figure", "fill", "final", "find", "fine",
    "finger", "finish", "fire", "first", "fish", "fit", "five", "flat", "floor", "flow", "flower",
    "fly", "follow", "food", "foot", "for", "force", "foreign", "forest", "forget", "form", "forward",
    "found", "four", "free", "fresh", "friend", "from", "front", "full", "fun", "game", "garden",
    "gas", "gate", "gather", "gave", "general", "get", "girl", "give", "glad", "glass", "go",
    "gold", "gone", "good", "got", "govern", "grand", "grass", "gray", "great", "green", "grew",
    "ground", "group", "grow", "guess", "guide", "gun", "had", "hair", "half", "hand", "hang",
    "happen", "happy", "hard", "has", "have", "he", "head", "hear", "heard", "heart", "heat",
    "heavy", "held", "help", "her", "here", "high", "hill", "him", "his", "history", "hit", "hold",
    "hole", "home", "hope", "horse", "hot", "hour", "house", "how", "huge", "human", "hundred",
    "hunt", "hurry", "hurt", "husband", "i", "ice", "idea", "if", "ill", "image", "imagine",
    "important", "in", "inch", "include", "increase", "indeed", "industry", "inside", "instead",
    "interest", "into", "iron", "is", "island", "issue", "it", "its", "job", "join", "joy", "judge",
    "jump", "just", "keep", "kept", "key", "kill", "kind", "king", "knew", "know", "labor", "lady",
    "laid", "lake", "land", "language", "large", "last", "late", "laugh", "law", "lay", "lead",
    "leader", "leaf", "learn", "least", "leave", "led", "left", "leg", "length", "less", "lesson",
    "let", "letter", "level", "lie", "life", "lift", "light", "like", "line", "lip", "liquid",
    "list", "listen", "little", "live", "load", "local", "locate", "lock", "long", "look", "lose",
    "loss", "lost", "lot", "loud", "love", "low", "machine", "made", "main", "major", "make", "man",
    "many", "map", "march", "mark", "market", "mass", "master", "match", "material", "matter",
    "may", "maybe", "me", "mean", "measure", "meat", "meet", "member", "men", "metal", "method",
    "middle", "might", "mile", "milk", "million", "mind", "mine", "minute", "miss", "mix", "modern",
    "moment", "money", "month", "moon", "more", "morning", "most", "mother", "motion", "mountain",
    "mouth", "move", "much", "music", "must", "my", "name", "nation", "natural", "nature", "near",
    "nearly", "necessary", "neck", "need", "neighbor", "neither", "nerve", "never", "new", "next",
    "night", "nine", "no", "noble", "noise", "none", "noon", "nor", "north", "nose", "not", "note",
    "nothing", "notice", "now", "number", "object", "observe", "obtain", "occur", "ocean", "of",
    "off", "offer", "office", "officer", "often", "oil", "old", "on", "once", "one", "only", "open",
    "operate", "opinion", "or", "order", "original", "other", "our", "out", "over", "own", "page",
    "pain", "paint", "pair", "paper", "part", "particular", "party", "pass", "past", "path", "pay",
    "peace", "people", "per", "perfect", "period", "person", "picture", "piece", "place", "plain",
    "plan", "plane", "planet", "plant", "play", "pleasant", "please", "plenty", "point", "police",
    "poor", "popular", "populate", "position", "possible", "post", "pound", "power", "practice",
    "prepare", "present", "president", "press", "pretty", "price", "print", "probable", "problem",
    "process", "produce", "product", "proper", "property", "protect", "proud", "prove", "provide",
    "public", "pull", "pure", "push", "put", "question", "quick", "quiet", "quite", "race", "radio",
    "railroad", "rain", "raise", "ran", "range", "rapid", "rather", "reach", "read", "ready",
    "real", "reason", "receive", "record", "red", "region", "remain", "remember", "repeat", "reply",
    "report", "represent", "require", "rest", "result", "return", "rich", "ride", "right", "ring",
    "rise", "river", "road", "rock", "roll", "room", "root", "rope", "rose", "round", "row", "rule",
    "run", "safe", "said", "sail", "same", "sand", "sat", "save", "saw", "say", "scale", "school",
    "science", "score", "sea", "season", "seat", "second", "secret", "section", "see", "seed",
    "seek", "seem", "seen", "self", "sell", "send", "sense", "sent", "sentence", "separate", "serve",
    "service", "set", "settle", "seven", "several", "shall", "shape", "share", "sharp", "she",
    "sheet", "shell", "shine", "ship", "shoe", "shook", "shoot", "shop", "shore", "short", "should",
    "shoulder", "shout", "show", "shut", "side", "sight", "sign", "signal", "silent", "silver",
    "similar", "simple", "since", "sing", "single", "sister", "sit", "six", "size", "skill", "skin",
    "sky", "sleep", "slow", "small", "smile", "smoke", "snow", "so", "soft", "soil", "soldier",
    "some", "son", "song", "soon", "sound", "south", "space", "speak", "special", "speed", "spell",
    "spend", "spoke", "spot", "spread", "spring", "square", "stage", "stand", "star", "start",
    "state", "station", "stay", "steam", "steel", "step", "stick", "still", "stone", "stood", "stop",
    "store", "story", "straight", "strange", "stream", "street", "strength", "strike", "strong",
    "student", "study", "subject", "substance", "such", "sudden", "suit", "summer", "sun", "supply",
    "support", "suppose", "sure", "surface", "surprise", "sweet", "system", "table", "tail", "take",
    "talk", "tall", "taste", "teach", "teacher", "team", "tell", "ten", "term", "test", "than",
    "thank", "that", "the", "their", "them", "then", "there", "therefore", "these", "they", "thick",
    "thin", "thing", "think", "third", "this", "those", "though", "thought", "thousand", "three",
    "through", "throw", "thus", "tie", "time", "tiny", "tire", "to", "today", "together", "told",
    "tomorrow", "tone", "too", "took", "tool", "top", "total", "touch", "toward", "town", "track",
    "trade", "train", "travel", "tree", "trip", "trouble", "true", "trust", "truth", "try", "turn",
    "twenty", "two", "type", "under", "understand", "unit", "until", "up", "upon", "us", "use",
    "usual", "valley", "value", "various", "verb", "very", "view", "village", "visit", "voice",
    "wait", "walk", "wall", "want", "war", "warm", "was", "wash", "watch", "water", "wave", "way",
    "we", "wear", "weather", "week", "weight", "well", "went", "were", "west", "what", "wheel",
    "when", "where", "whether", "which", "while", "white", "who", "whole", "whose", "why", "wide",
    "wife", "wild", "will", "win", "wind", "window", "winter", "wire", "wise", "wish", "with",
    "without", "woman", "women", "wonder", "wood", "word", "work", "world", "worth", "would",
    "write", "wrong", "yard", "year", "yellow", "yes", "yesterday", "yet", "you", "young", "your",
];

/// Coxhead Academic Word List (AWL) primary roots.
const ACADEMIC_WORD_LIST_ROOTS: &[&str] = &[
    "analy", "approach", "assess", "assum", "author", "avail", "benefit", "concept", "consist",
    "constitut", "context", "contract", "creat", "data", "defin", "deriv", "distribut", "econom",
    "environ", "establish", "estimat", "evidenc", "export", "factor", "financ", "formula", "function",
    "identif", "incom", "indicat", "individu", "interpret", "involv", "issu", "labor", "legal",
    "legislat", "major", "method", "occur", "percent", "period", "policy", "principl", "proceed",
    "process", "requir", "research", "respond", "role", "section", "sector", "significan", "similar",
    "sourc", "specific", "structur", "theor", "vary", "achiev", "acquir", "administ", "affect",
    "appropriat", "aspect", "assist", "categor", "chapter", "commiss", "communit", "complex",
    "comput", "conclud", "conduct", "consequenc", "construct", "consum", "credit", "cultur", "design",
    "distinct", "element", "equat", "evaluat", "featur", "final", "focus", "impact", "injur",
    "institut", "invest", "item", "journal", "maintain", "normal", "obtain", "participat", "perceiv",
    "positiv", "potenti", "previous", "primar", "purchas", "rang", "region", "regulat", "relevan",
    "resid", "resourc", "restrict", "secur", "sought", "select", "site", "strateg", "survey", "text",
    "tradition", "transfer", "alternat", "circumstanc", "comment", "compensat", "compon", "consent",
    "consider", "constant", "constrain", "contribut", "conven", "coordinat", "core", "corporat",
    "correspond", "criteria", "deduc", "demonstrat", "document", "dominat", "emphas", "ensur",
    "exclud", "framework", "fund", "illustrat", "immigrat", "impli", "initial", "instanc", "interact",
    "justif", "layer", "link", "locat", "maximiz", "minor", "negat", "outcom", "partner", "philosoph",
    "physic", "proport", "publish", "react", "regist", "reli", "scheme", "sequenc", "shift",
    "specif", "suffic", "task", "technic", "technolog", "valid", "volum", "apparent", "approximat",
    "attitud", "attribut", "civil", "code", "commit", "communicat", "concentrat", "confer", "contrast",
    "cycl", "debate", "despit", "dimens", "domest", "emerg", "error", "ethnic", "goal", "grant",
    "hypothes", "implement", "implicat", "impos", "integrat", "intern", "investig", "mechan",
    "occup", "output", "overal", "parallel", "paramet", "phase", "predict", "prior", "profession",
    "project", "promot", "regim", "resolv", "retain", "seri", "statist", "status", "stress",
    "subsequent", "sum", "summar", "undertak",
];

/// Deterministic Pedagogical & Reading Level Scorer.
pub struct PedagogicalScorer {
    dale_chall_set: HashSet<&'static str>,
}

impl Default for PedagogicalScorer {
    fn default() -> Self {
        let mut set = HashSet::with_capacity(DALE_CHALL_COMMON_LEMMAS.len());
        for &w in DALE_CHALL_COMMON_LEMMAS {
            set.insert(w);
        }
        Self { dale_chall_set: set }
    }
}

impl PedagogicalScorer {
    /// Counts syllables in an English word using deterministic vowel-cluster heuristics.
    pub fn count_syllables(word: &str) -> usize {
        let clean: String = word
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect();

        if clean.len() <= 3 {
            return 1;
        }

        let bytes = clean.as_bytes();
        let mut syllables = 0usize;
        let mut prev_vowel = false;

        let is_vowel = |b: u8| -> bool {
            b == b'a' || b == b'e' || b == b'i' || b == b'o' || b == b'u' || b == b'y'
        };

        for &b in bytes {
            let vowel = is_vowel(b);
            if vowel && !prev_vowel {
                syllables += 1;
            }
            prev_vowel = vowel;
        }

        // Silent 'e' at end
        if bytes.last() == Some(&b'e') && bytes.len() > 2 && !is_vowel(bytes[bytes.len() - 2]) {
            syllables = syllables.saturating_sub(1);
        }
        // '-ed' or '-es' endings
        if clean.ends_with("ed") || clean.ends_with("es") {
            if syllables > 1 {
                syllables = syllables.saturating_sub(1);
            }
        }

        syllables.max(1)
    }

    /// Evaluates reading level, Flesch-Kincaid grade, and SmolLM2 curriculum tier.
    pub fn score_document(&self, text: &str) -> PedagogicalScore {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return PedagogicalScore {
                fkgl_grade_level: 0.0,
                flesch_reading_ease: 100.0,
                dale_chall_score: 0.0,
                academic_word_density: 0.0,
                words_count: 0,
                sentences_count: 0,
                syllables_count: 0,
                complex_words_ratio: 0.0,
                recommended_stage: CurriculumStage::Stage1Fundamental,
                pedagogical_summary: "Empty text".to_string(),
            };
        }

        let words: Vec<&str> = trimmed
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty())
            .collect();

        let sentences: Vec<&str> = trimmed
            .split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let n_words = words.len().max(1);
        let n_sentences = sentences.len().max(1);

        let mut total_syllables = 0usize;
        let mut complex_words = 0usize;
        let mut unfamiliar_words = 0usize;
        let mut academic_words = 0usize;

        for &w in &words {
            let lower = w.to_lowercase();
            let syl = Self::count_syllables(&lower);
            total_syllables += syl;

            if syl >= 3 {
                complex_words += 1;
            }

            // Dale-Chall familiarity check
            if !self.dale_chall_set.contains(lower.as_str()) {
                unfamiliar_words += 1;
            }

            // Academic Word List (AWL) root match
            for &root in ACADEMIC_WORD_LIST_ROOTS {
                if lower.starts_with(root) {
                    academic_words += 1;
                    break;
                }
            }
        }

        let words_f = n_words as f64;
        let sent_f = n_sentences as f64;
        let syl_f = total_syllables as f64;

        // 1. Flesch-Kincaid Grade Level (FKGL)
        let fkgl = 0.39 * (words_f / sent_f) + 11.8 * (syl_f / words_f) - 15.59;
        let fkgl_clamped = fkgl.clamp(1.0, 30.0);

        // 2. Flesch Reading Ease (FRE)
        let fre = 206.835 - 1.015 * (words_f / sent_f) - 84.6 * (syl_f / words_f);

        // 3. Dale-Chall Readability Index
        let unfamiliar_ratio = (unfamiliar_words as f64 / words_f) * 100.0;
        let mut dale_chall = 0.1579 * unfamiliar_ratio + 0.0496 * (words_f / sent_f);
        if unfamiliar_ratio > 5.0 {
            dale_chall += 3.6365; // standard adjustment
        }

        // 4. Academic Word List Density
        let awl_density = academic_words as f64 / words_f;
        let complex_ratio = complex_words as f64 / words_f;

        // 5. SmolLM2 Curriculum Routing
        let recommended_stage = if fkgl_clamped > 16.0 || awl_density >= 0.12 {
            CurriculumStage::Stage4Research
        } else if fkgl_clamped > 12.0 || awl_density >= 0.07 {
            CurriculumStage::Stage3Advanced
        } else if fkgl_clamped > 8.0 || awl_density >= 0.035 {
            CurriculumStage::Stage2Intermediate
        } else {
            CurriculumStage::Stage1Fundamental
        };

        let summary = format!(
            "FKGL Grade {:.1} (FRE {:.0}) | Dale-Chall {:.1} | AWL Density {:.1}% | Assigned -> {}",
            fkgl_clamped,
            fre,
            dale_chall,
            awl_density * 100.0,
            recommended_stage.as_str()
        );

        PedagogicalScore {
            fkgl_grade_level: fkgl_clamped,
            flesch_reading_ease: fre,
            dale_chall_score: dale_chall,
            academic_word_density: awl_density,
            words_count: n_words,
            sentences_count: n_sentences,
            syllables_count: total_syllables,
            complex_words_ratio: complex_ratio,
            recommended_stage,
            pedagogical_summary: summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementary_prose_mapped_to_stage1() {
        let scorer = PedagogicalScorer::default();
        let simple_text = "The dog ran across the green grass. The boy saw the ball and ran to get it. They played all day until the sun went down.";
        let score = scorer.score_document(simple_text);
        assert!(score.fkgl_grade_level <= 6.0, "Expected low grade level, got {}", score.fkgl_grade_level);
        assert_eq!(score.recommended_stage, CurriculumStage::Stage1Fundamental);
    }

    #[test]
    fn test_scientific_academic_prose_mapped_to_stage3_or_stage4() {
        let scorer = PedagogicalScorer::default();
        let research_text = "The empirical methodology incorporates longitudinal cohort regression analyses to establish parametric causality. The structural hypothesis demonstrates significant correlation within the multivariate econometric framework.";
        let score = scorer.score_document(research_text);
        assert!(score.fkgl_grade_level >= 13.0, "Expected graduate level, got {}", score.fkgl_grade_level);
        assert!(score.academic_word_density > 0.10);
        assert!(matches!(score.recommended_stage, CurriculumStage::Stage3Advanced | CurriculumStage::Stage4Research));
    }
}
