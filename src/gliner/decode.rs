//! Pure span decoding — port of `gliner.decoding.SpanDecoder` (`_decode_batch` + `greedy_search`).
//! Kept free of ONNX types so it is unit-testable and mirrors the Lean model in `lean/GLinker.lean`.

use std::collections::HashSet;
use ndarray::ArrayView3;

#[derive(Clone, Debug, PartialEq)]
pub struct RawSpan {
    /// inclusive word indices
    pub start: usize,
    pub end: usize,
    pub class: usize,
    pub score: f32,
    /// (class index, prob), descending
    pub class_probs: Vec<(usize, f32)>,
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode one item. `logits` has shape (L, K, C) (words, widths, classes).
/// - `filter`: GLinker `input_spans` as word spans (start, end) inclusive; `None` = all spans.
/// - keeps `prob > threshold` (strict, as in Python), and `start + width + 1 <= num_words`.
pub fn decode_item(
    logits: ArrayView3<f32>,
    num_words: usize,
    filter: Option<&HashSet<(usize, usize)>>,
    threshold: f32,
    top_k: usize,
) -> Vec<RawSpan> {
    let (l, k, c) = logits.dim();
    let mut out = Vec::new();
    for s in 0..l.min(num_words) {
        for w in 0..k {
            let e = s + w;
            if e + 1 > num_words {
                break;
            }
            if let Some(f) = filter {
                if !f.contains(&(s, e)) {
                    continue;
                }
            }
            let mut probs: Option<Vec<(usize, f32)>> = None;
            for cls in 0..c {
                let p = sigmoid(logits[[s, w, cls]]);
                if p > threshold {
                    let cp = probs
                        .get_or_insert_with(|| {
                            let mut v: Vec<(usize, f32)> = (0..c).map(|j| (j, sigmoid(logits[[s, w, j]]))).collect();
                            v.sort_by(|a, b| b.1.total_cmp(&a.1));
                            v.truncate(top_k);
                            v
                        })
                        .clone();
                    out.push(RawSpan { start: s, end: e, class: cls, score: p, class_probs: cp });
                }
            }
        }
    }
    out
}

/// `has_overlapping` (flat) / `has_overlapping_nested` (nested), with the multi-label same-span rule.
pub fn conflicts(a: &RawSpan, b: &RawSpan, flat_ner: bool, multi_label: bool) -> bool {
    if a.start == b.start && a.end == b.end {
        return !multi_label;
    }
    let disjoint = a.start > b.end || b.start > a.end;
    if flat_ner {
        !disjoint
    } else {
        let nested = (a.start <= b.start && a.end >= b.end) || (b.start <= a.start && b.end >= a.end);
        !(disjoint || nested)
    }
}

/// Greedy search: highest score first, keep a span iff it conflicts with no kept span;
/// result ordered by (start, end). Proven pairwise conflict-free in Lean (`greedy_pairwise`).
pub fn greedy(mut spans: Vec<RawSpan>, flat_ner: bool, multi_label: bool) -> Vec<RawSpan> {
    spans.sort_by(|a, b| b.score.total_cmp(&a.score)); // stable, like Python `sorted`
    let mut kept: Vec<RawSpan> = Vec::new();
    for s in spans {
        if !kept.iter().any(|k| conflicts(k, &s, flat_ner, multi_label)) {
            kept.push(s);
        }
    }
    kept.sort_by_key(|s| (s.start, s.end));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn logit(p: f32) -> f32 { (p / (1.0 - p)).ln() }

    #[test]
    fn filter_threshold_greedy() {
        // 3 words, K=2, C=2
        let mut a = Array3::from_elem((3, 2, 2), -20.0f32);
        a[[0, 0, 0]] = logit(0.9); // (0,0) c0
        a[[0, 1, 1]] = logit(0.8); // (0,1) c1 overlaps (0,0)
        a[[2, 0, 1]] = logit(0.7); // (2,2) c1
        a[[2, 1, 0]] = logit(0.99); // (2,3) invalid: beyond num_words
        let all = decode_item(a.view(), 3, None, 0.5, 5);
        assert_eq!(all.len(), 3);
        let g = greedy(all, true, false);
        assert_eq!(g.iter().map(|s| (s.start, s.end, s.class)).collect::<Vec<_>>(), vec![(0, 0, 0), (2, 2, 1)]);
        // input-span constraint keeps only (0,1)
        let f: HashSet<_> = [(0usize, 1usize)].into_iter().collect();
        let c = decode_item(a.view(), 3, Some(&f), 0.5, 5);
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].start, c[0].end, c[0].class), (0, 1, 1));
        assert_eq!(c[0].class_probs[0].0, 1);
    }

    #[test]
    fn nested_mode_allows_nesting() {
        let mk = |s, e, sc| RawSpan { start: s, end: e, class: 0, score: sc, class_probs: vec![] };
        let g = greedy(vec![mk(0, 3, 0.9), mk(1, 2, 0.8), mk(2, 5, 0.7)], false, false);
        assert_eq!(g.len(), 2); // (2,5) crosses (0,3)
    }
}
