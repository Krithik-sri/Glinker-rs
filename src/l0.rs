//! L0 — aggregation of L1 mentions, L2 candidates and L3/L4 links (Python `L0Component`).
//!
//! Divergences from Python, both deliberate:
//! * a mention's candidates are the L2 results *for that mention* (Python re-derives them by
//!   exact label/alias equality, which drops fuzzy hits);
//! * the linked `entity_id` comes straight from L3's label→candidate map instead of
//!   re-formatting templates plus a substring fallback.

use crate::template;
use crate::types::*;

#[derive(Clone, Debug)]
pub struct L0Options {
    pub min_confidence: f32,
    pub include_unlinked: bool,
    pub strict_matching: bool,
    pub position_tolerance: usize,
}
impl Default for L0Options {
    fn default() -> Self {
        Self { min_confidence: 0.0, include_unlinked: true, strict_matching: true, position_tolerance: 2 }
    }
}

fn scores(p: &SpanPrediction, cands: &[DatabaseRecord], tpl: &str) -> Vec<(String, f32)> {
    p.class_probs
        .iter()
        .filter_map(|(label, prob)| {
            let l = label.trim().to_lowercase();
            cands.iter().find(|c| template::format(tpl, c).trim().to_lowercase() == l).map(|c| (c.entity_id.clone(), *prob))
        })
        .collect()
}

fn linked(p: &SpanPrediction, cands: &[DatabaseRecord]) -> LinkedEntity {
    let id = p.entity_id.clone().unwrap_or_else(|| "unknown".into());
    let label = cands.iter().find(|c| c.entity_id == id).map(|c| c.label.clone()).unwrap_or_else(|| p.label.clone());
    LinkedEntity { entity_id: id, label, confidence: p.score, start: p.start, end: p.end, matched_text: p.text.clone() }
}

fn dist(a: usize, b: usize) -> usize {
    a.abs_diff(b)
}

/// Aggregate one text.
pub fn aggregate_text(
    mentions: &[Mention],
    per_mention_candidates: &[Vec<DatabaseRecord>],
    text_candidates: &[DatabaseRecord],
    links: &[SpanPrediction],
    tpl: &str,
    o: &L0Options,
) -> Vec<L0Entity> {
    let mut used = vec![false; links.len()];
    let mut out = Vec::new();
    for (mi, m) in mentions.iter().enumerate() {
        let cands = per_mention_candidates.get(mi).cloned().unwrap_or_default();
        let hit = links
            .iter()
            .position(|p| p.start == m.start && p.end == m.end)
            .or_else(|| links.iter().position(|p| dist(p.start, m.start) <= o.position_tolerance && dist(p.end, m.end) <= o.position_tolerance));
        let (le, cs) = match hit {
            Some(i) => {
                used[i] = true;
                (Some(linked(&links[i], text_candidates)), scores(&links[i], text_candidates, tpl))
            }
            None => (None, vec![]),
        };
        let stage = if le.is_some() { Stage::L3Linked } else if !cands.is_empty() { Stage::L2Found } else { Stage::L1Only };
        out.push(L0Entity {
            mention_text: m.text.clone(),
            label: m.label.clone(),
            mention_start: m.start,
            mention_end: m.end,
            left_context: m.left_context.clone(),
            right_context: m.right_context.clone(),
            num_candidates: cands.len(),
            candidates: cands,
            is_linked: le.is_some(),
            linked_entity: le,
            candidate_scores: cs,
            pipeline_stage: stage,
        });
    }
    if !o.strict_matching {
        for (i, p) in links.iter().enumerate() {
            if used[i] {
                continue;
            }
            let le = linked(p, text_candidates);
            let cand: Vec<DatabaseRecord> = text_candidates.iter().filter(|c| c.entity_id == le.entity_id).take(1).cloned().collect();
            out.push(L0Entity {
                mention_text: p.text.clone(),
                label: None,
                mention_start: p.start,
                mention_end: p.end,
                left_context: String::new(),
                right_context: String::new(),
                num_candidates: cand.len(),
                candidates: cand,
                is_linked: true,
                linked_entity: Some(le),
                candidate_scores: scores(p, text_candidates, tpl),
                pipeline_stage: Stage::L3Only,
            });
        }
    }
    // filter_by_confidence → sort_by_confidence (stable, descending)
    out.retain(|e| match &e.linked_entity {
        Some(l) => l.confidence >= o.min_confidence,
        None => o.include_unlinked,
    });
    out.sort_by(|a, b| {
        let ca = a.linked_entity.as_ref().map_or(0.0, |l| l.confidence);
        let cb = b.linked_entity.as_ref().map_or(0.0, |l| l.confidence);
        cb.total_cmp(&ca)
    });
    out
}

pub fn stats(entities: &[Vec<L0Entity>]) -> Stats {
    let mut s = Stats::default();
    for e in entities.iter().flatten() {
        s.total_mentions += 1;
        if e.is_linked { s.linked += 1 } else { s.unlinked += 1 }
        match e.pipeline_stage {
            Stage::L1Only => s.l1_only += 1,
            Stage::L2Found => s.l2_found += 1,
            Stage::L3Linked => s.l3_linked += 1,
            Stage::L3Only => s.l3_only += 1,
        }
    }
    s.linking_rate = if s.total_mentions > 0 { s.linked as f64 / s.total_mentions as f64 } else { 0.0 };
    s
}
