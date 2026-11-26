use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Fast 64-bit FNV-1a hash for string tokens.
#[inline]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Disjoint-Set Union (Union-Find) for connected component clustering.
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        let mut curr = i;
        while curr != root {
            let next = self.parent[curr];
            self.parent[curr] = root;
            curr = next;
        }
        root
    }

    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i == root_j {
            return false;
        }
        if self.rank[root_i] < self.rank[root_j] {
            self.parent[root_i] = root_j;
        } else if self.rank[root_i] > self.rank[root_j] {
            self.parent[root_j] = root_i;
        } else {
            self.parent[root_j] = root_i;
            self.rank[root_i] += 1;
        }
        true
    }
}

/// MinHash LSH Configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashLshConfig {
    /// Number of universal hash permutations (typically 128).
    pub num_hashes: usize,
    /// Shingle word length k (default 5).
    pub shingle_size: usize,
    /// Number of LSH bands (b). Default: 16.
    pub num_bands: usize,
    /// Number of rows per band (r). Default: 8 (num_hashes = num_bands * rows_per_band = 128).
    pub rows_per_band: usize,
    /// Minimum Jaccard similarity threshold for pairwise deduplication confirmation. Default: 0.80.
    pub jaccard_threshold: f64,
}

impl Default for MinHashLshConfig {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            shingle_size: 5,
            num_bands: 16,
            rows_per_band: 8,
            jaccard_threshold: 0.80,
        }
    }
}

/// Summary report of MinHash LSH deduplication run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationSummary {
    pub total_documents: usize,
    pub unique_documents_kept: usize,
    pub duplicate_documents_purged: usize,
    pub duplicate_clusters_count: usize,
    pub retention_ratio: f64,
}

/// High-performance MinHash LSH Deduplication Engine.
pub struct MinHashLshEngine {
    pub config: MinHashLshConfig,
    /// Coefficients (a_i, b_i) for universal hashing h_i(x) = (a_i * x + b_i).
    hash_coeffs: Vec<(u64, u64)>,
}

impl MinHashLshEngine {
    pub fn new(config: MinHashLshConfig) -> Self {
        assert_eq!(
            config.num_hashes,
            config.num_bands * config.rows_per_band,
            "num_hashes must equal num_bands * rows_per_band"
        );

        // Precompute deterministic pseudorandom universal hash coefficients
        let mut coeffs = Vec::with_capacity(config.num_hashes);
        let mut seed: u64 = 0x85ebca6b;
        for _ in 0..config.num_hashes {
            // Linear congruential generator for seeds
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = seed | 1; // ensure odd
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = seed;
            coeffs.push((a, b));
        }

        Self {
            config,
            hash_coeffs: coeffs,
        }
    }

    /// Generates word k-shingles from text and hashes them to 64-bit integers.
    pub fn extract_shingles(&self, text: &str) -> Vec<u64> {
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return Vec::new();
        }

        let k = self.config.shingle_size;
        if words.len() < k {
            // For brief text, use single-word hashes or character shingles
            return words.iter().map(|w| fnv1a_64(w.to_lowercase().as_bytes())).collect();
        }

        let mut shingles = Vec::with_capacity(words.len() - k + 1);
        let mut buf = String::with_capacity(64);
        for i in 0..=(words.len() - k) {
            buf.clear();
            for j in 0..k {
                if j > 0 {
                    buf.push(' ');
                }
                buf.push_str(&words[i + j].to_lowercase());
            }
            shingles.push(fnv1a_64(buf.as_bytes()));
        }
        shingles
    }

    /// Computes m-dimensional MinHash signature vector for a document.
    pub fn compute_signature(&self, text: &str) -> Vec<u64> {
        let shingles = self.extract_shingles(text);
        if shingles.is_empty() {
            return vec![u64::MAX; self.config.num_hashes];
        }

        let mut signature = vec![u64::MAX; self.config.num_hashes];
        for &s in &shingles {
            for (idx, &(a, b)) in self.hash_coeffs.iter().enumerate() {
                let h = a.wrapping_mul(s).wrapping_add(b);
                if h < signature[idx] {
                    signature[idx] = h;
                }
            }
        }
        signature
    }

    /// Estimates exact Jaccard similarity between two MinHash signatures.
    pub fn signature_jaccard_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
        assert_eq!(sig1.len(), sig2.len());
        let matches = sig1.iter().zip(sig2.iter()).filter(|(&a, &b)| a == b).count();
        matches as f64 / sig1.len() as f64
    }

    /// Computes true Jaccard similarity between two texts using exact shingle set intersection.
    pub fn exact_jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let s1: HashSet<u64> = self.extract_shingles(text1).into_iter().collect();
        let s2: HashSet<u64> = self.extract_shingles(text2).into_iter().collect();
        if s1.is_empty() && s2.is_empty() {
            return 1.0;
        }
        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }
        let intersection = s1.intersection(&s2).count();
        let union = s1.union(&s2).count();
        intersection as f64 / union as f64
    }

    /// Parallel deduplication across a slice of document texts.
    /// Returns boolean mask: true = keep unique document, false = duplicate to purge.
    pub fn deduplicate_texts(&self, docs: &[&str]) -> (Vec<bool>, DeduplicationSummary) {
        let n = docs.len();
        if n == 0 {
            return (
                Vec::new(),
                DeduplicationSummary {
                    total_documents: 0,
                    unique_documents_kept: 0,
                    duplicate_documents_purged: 0,
                    duplicate_clusters_count: 0,
                    retention_ratio: 1.0,
                },
            );
        }

        // 1. Parallel signature generation
        let signatures: Vec<Vec<u64>> = docs
            .par_iter()
            .map(|text| self.compute_signature(text))
            .collect();

        // 2. LSH Band Bucketing
        let b = self.config.num_bands;
        let r = self.config.rows_per_band;

        // Band hash buckets: (band_idx, band_hash) -> Vec<doc_id>
        let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
        for (doc_id, sig) in signatures.iter().enumerate() {
            for band_idx in 0..b {
                let start = band_idx * r;
                let end = start + r;
                let band_slice = &sig[start..end];
                // Hash the band slice
                let band_hash = fnv1a_64(bytemuck_slice(band_slice));
                buckets.entry((band_idx, band_hash)).or_default().push(doc_id);
            }
        }

        // 3. Candidate Pair Verification & DSU Clustering
        let mut dsu = DisjointSet::new(n);
        let mut verified_pairs = HashSet::new();

        for bucket in buckets.values() {
            if bucket.len() < 2 {
                continue;
            }
            // Check candidate pairs in bucket
            for i in 0..bucket.len() {
                for j in (i + 1)..bucket.len() {
                    let u = bucket[i];
                    let v = bucket[j];
                    let pair_key = (u.min(v), u.max(v));
                    if verified_pairs.contains(&pair_key) {
                        continue;
                    }
                    verified_pairs.insert(pair_key);

                    // Check MinHash signature similarity first
                    let sig_sim = Self::signature_jaccard_similarity(&signatures[u], &signatures[v]);
                    if sig_sim >= (self.config.jaccard_threshold - 0.08) {
                        // Confirm with exact Jaccard similarity
                        let exact_sim = self.exact_jaccard_similarity(docs[u], docs[v]);
                        if exact_sim >= self.config.jaccard_threshold {
                            dsu.union(u, v);
                        }
                    }
                }
            }
        }

        // 4. Resolve clusters: Keep the longest document in each cluster
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for doc_id in 0..n {
            let root = dsu.find(doc_id);
            clusters.entry(root).or_default().push(doc_id);
        }

        let mut keep_mask = vec![false; n];
        let mut duplicate_clusters = 0usize;
        let mut kept_count = 0usize;

        for members in clusters.values() {
            if members.len() > 1 {
                duplicate_clusters += 1;
                // Elect longest document as winner
                let best_doc_id = *members
                    .iter()
                    .max_by_key(|&&id| docs[id].len())
                    .unwrap();
                keep_mask[best_doc_id] = true;
                kept_count += 1;
            } else {
                keep_mask[members[0]] = true;
                kept_count += 1;
            }
        }

        let purged_count = n - kept_count;
        let retention = kept_count as f64 / n as f64;

        (
            keep_mask,
            DeduplicationSummary {
                total_documents: n,
                unique_documents_kept: kept_count,
                duplicate_documents_purged: purged_count,
                duplicate_clusters_count: duplicate_clusters,
                retention_ratio: retention,
            },
        )
    }
}

/// Helper to view u64 slice as byte slice.
fn bytemuck_slice(u64s: &[u64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            u64s.as_ptr() as *const u8,
            u64s.len() * std::mem::size_of::<u64>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_identical_documents_yield_similarity_one() {
        let engine = MinHashLshEngine::new(MinHashLshConfig::default());
        let text1 = "The solar system consists of the Sun and the astronomical objects bound in orbit around it.";
        let text2 = "The solar system consists of the Sun and the astronomical objects bound in orbit around it.";
        let sig1 = engine.compute_signature(text1);
        let sig2 = engine.compute_signature(text2);
        let sim = MinHashLshEngine::signature_jaccard_similarity(&sig1, &sig2);
        assert!((sim - 1.0).abs() < 1e-6);
        assert_eq!(engine.exact_jaccard_similarity(text1, text2), 1.0);
    }

    #[test]
    fn test_minhash_near_duplicates_clustered_and_purged() {
        let engine = MinHashLshEngine::new(MinHashLshConfig::default());
        let doc1 = "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at atomic and subatomic scales.";
        let doc2 = "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at atomic and subatomic scales! Published on June 2024."; // Near duplicate with small suffix
        let doc3 = "The French Revolution was a period of political and societal change in France that began with the Estates General of 1789."; // Completely distinct

        let docs = vec![doc1, doc2, doc3];
        let (mask, summary) = engine.deduplicate_texts(&docs);

        assert_eq!(summary.total_documents, 3);
        assert_eq!(summary.unique_documents_kept, 2);
        assert_eq!(summary.duplicate_documents_purged, 1);
        assert_eq!(summary.duplicate_clusters_count, 1);

        // doc3 must be kept
        assert!(mask[2]);
        // Out of doc1 and doc2, the longer one (doc2) should be kept
        assert!(mask[1]);
        assert!(!mask[0]);
    }
}
