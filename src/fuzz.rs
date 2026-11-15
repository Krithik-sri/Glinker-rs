//! Port of `rapidfuzz.fuzz.ratio` (normalized Indel similarity), used by the L2 dict layer.
//!
//! `ratio(a, b) = 1 - indel(a, b) / (|a| + |b|)` where `indel = |a| + |b| - 2·LCS(a, b)`,
//! i.e. `ratio = 2·LCS / (|a| + |b|)`; both empty → 1.0.
//! Lean4 (`lean/GLinker.lean`): `lcs_le_min` gives 0 ≤ ratio ≤ 1, and `lcs_comm` gives symmetry.

/// Longest common subsequence length over chars, O(|a|·|b|) time, O(|b|) memory.
pub fn lcs(a: &[char], b: &[char]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for &x in a {
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { prev[j + 1].max(cur[j]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// `rapidfuzz.fuzz.ratio(a, b) / 100`.
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * lcs(&a, &b) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_rapidfuzz() {
        // rapidfuzz.fuzz.ratio("this is a test", "this is a test!") == 96.55172413793103
        assert!((ratio("this is a test", "this is a test!") - 0.9655172413793103).abs() < 1e-12);
        // fuzz.ratio("insulin", "insuline") == 93.33333333333333
        assert!((ratio("insulin", "insuline") - 0.9333333333333333).abs() < 1e-12);
        assert_eq!(ratio("", ""), 1.0);
        assert_eq!(ratio("abc", ""), 0.0);
        assert_eq!(ratio("ab", "ba"), ratio("ba", "ab"));
    }
}
