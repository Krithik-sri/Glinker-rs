//! Matthews Correlation Coefficient for entity-linking evaluation.
//!
//! * [`mcc_binary`] — candidate-level: every (mention, candidate) pair is a binary decision
//!   "is this the linked entity?" vs. gold. TNs exist naturally (rejected wrong candidates).
//! * [`mcc_multiclass`] — Gorodkin R_K over mention-level labels (entity_id or `NIL`), the
//!   multiclass generalisation used by scikit-learn's `matthews_corrcoef`.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::types::L0Entity;

pub const NIL: &str = "NIL";

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Confusion {
    pub tp: u64,
    pub tn: u64,
    pub fp: u64,
    pub fn_: u64,
}

impl Confusion {
    pub fn add(&mut self, truth: bool, pred: bool) {
        match (truth, pred) {
            (true, true) => self.tp += 1,
            (false, false) => self.tn += 1,
            (false, true) => self.fp += 1,
            (true, false) => self.fn_ += 1,
        }
    }
}

/// (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)); 0 when any marginal is 0 (sklearn convention).
pub fn mcc_binary(c: &Confusion) -> f64 {
    let (tp, tn, fp, fn_) = (c.tp as f64, c.tn as f64, c.fp as f64, c.fn_ as f64);
    let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if den == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / den }
}

/// Gorodkin multiclass MCC over (truth, pred) label pairs.
pub fn mcc_multiclass(pairs: &[(String, String)]) -> f64 {
    let s = pairs.len() as f64;
    if pairs.is_empty() {
        return 0.0;
    }
    let mut t: BTreeMap<&str, f64> = BTreeMap::new();
    let mut p: BTreeMap<&str, f64> = BTreeMap::new();
    let mut c = 0.0;
    for (a, b) in pairs {
        *t.entry(a).or_default() += 1.0;
        *p.entry(b).or_default() += 1.0;
        if a == b { c += 1.0; }
    }
    let keys: BTreeSet<&str> = t.keys().chain(p.keys()).copied().collect();
    let pt: f64 = keys.iter().map(|k| p.get(k).unwrap_or(&0.0) * t.get(k).unwrap_or(&0.0)).sum();
    let pp: f64 = p.values().map(|x| x * x).sum();
    let tt: f64 = t.values().map(|x| x * x).sum();
    let den = ((s * s - pp) * (s * s - tt)).sqrt();
    if den == 0.0 { 0.0 } else { (c * s - pt) / den }
}

/// Gold annotation line: `{"text": "...", "entities": [{"start":0,"end":4,"entity_id":"Q1"}]}`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GoldDoc {
    pub text: String,
    pub entities: Vec<GoldSpan>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GoldSpan {
    pub start: usize,
    pub end: usize,
    pub entity_id: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EvalReport {
    pub docs: usize,
    pub gold_mentions: usize,
    pub predicted_links: usize,
    pub candidate_confusion: Confusion,
    pub mcc_candidate: f64,
    pub mcc_mention: f64,
    pub accuracy_mention: f64,
}

/// Score L0 output against gold. Mention items = union of gold spans and predicted spans.
pub fn evaluate(gold: &[GoldDoc], pred: &[Vec<L0Entity>]) -> EvalReport {
    let mut r = EvalReport { docs: gold.len(), ..Default::default() };
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (g, p) in gold.iter().zip(pred) {
        r.gold_mentions += g.entities.len();
        let mut gmap: BTreeMap<(usize, usize), String> = g.entities.iter().map(|e| ((e.start, e.end), e.entity_id.clone())).collect();
        for e in p {
            let span = (e.mention_start, e.mention_end);
            let linked = e.linked_entity.as_ref().map(|l| l.entity_id.clone()).unwrap_or_else(|| NIL.into());
            if linked != NIL { r.predicted_links += 1; }
            let truth = gmap.remove(&span).unwrap_or_else(|| NIL.into());
            // candidate-level binary decisions
            // decision set = L2 candidates ∪ every candidate L3 scored (class_probs) ∪ the link
            let mut ids: BTreeSet<String> = e.candidates.iter().map(|c| c.entity_id.clone()).collect();
            ids.extend(e.candidate_scores.iter().map(|(id, _)| id.clone()));
            if linked != NIL { ids.insert(linked.clone()); }
            for id in ids {
                r.candidate_confusion.add(id == truth, id == linked);
            }
            pairs.push((truth, linked));
        }
        for (_, id) in gmap {
            // gold mention never surfaced by the pipeline: a missed link
            r.candidate_confusion.add(true, false);
            pairs.push((id, NIL.into()));
        }
    }
    r.mcc_candidate = mcc_binary(&r.candidate_confusion);
    r.mcc_mention = mcc_multiclass(&pairs);
    r.accuracy_mention = if pairs.is_empty() { 0.0 } else { pairs.iter().filter(|(a, b)| a == b).count() as f64 / pairs.len() as f64 };
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_matches_sklearn() {
        // sklearn: y_true=[1,1,1,0,0,0,0], y_pred=[1,1,0,0,0,1,0] → 0.4166666666666667
        let c = Confusion { tp: 2, fn_: 1, fp: 1, tn: 3 };
        assert!((mcc_binary(&c) - 0.4166666666666667).abs() < 1e-12);
    }
    #[test]
    fn multiclass_matches_sklearn() {
        // sklearn: matthews_corrcoef([0,1,2,0,1,2],[0,2,1,0,0,1]) == 0.0
        let v: Vec<(String, String)> = [(0, 0), (1, 2), (2, 1), (0, 0), (1, 0), (2, 1)].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        assert!(mcc_multiclass(&v).abs() < 1e-12);
        // sklearn: matthews_corrcoef(list("aabbcca"), list("abbccaa")) == 0.34375
        let w: Vec<(String, String)> = "aabbcca".chars().zip("abbccaa".chars()).map(|(a, b)| (a.to_string(), b.to_string())).collect();
        assert!((mcc_multiclass(&w) - 0.34375).abs() < 1e-12);
        let perfect: Vec<(String, String)> = ["a", "b", "c"].iter().map(|x| (x.to_string(), x.to_string())).collect();
        assert!((mcc_multiclass(&perfect) - 1.0).abs() < 1e-12);
    }
}
