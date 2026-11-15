//! L4 — uni-encoder reranker with candidate chunking (Python `L4Processor`/`L4Component`).
//! Labels are split into chunks of `max_labels`, each chunk scored with the L1 `input_spans`,
//! then: deduplicate (best score per span) → filter_by_score → sort_by_position.

use crate::gliner::engine::SpanModel;
use crate::gliner::pipeline::SpanInput;
use crate::l3::input_spans;
use crate::template;
use crate::types::{DatabaseRecord, Mention, SpanPrediction};
use crate::Result;

pub struct L4 {
    pub model: SpanModel,
    pub template: String,
    pub max_labels: usize,
}

impl L4 {
    pub fn rerank(&self, texts: &[String], candidates: &[Vec<DatabaseRecord>], mentions: Option<&[Vec<Mention>]>, shared: bool) -> Result<Vec<Vec<SpanPrediction>>> {
        let spans = input_spans(mentions, texts.len());
        let mut out = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let cands = if shared { &candidates[0] } else { candidates.get(i).map(Vec::as_slice).unwrap_or(&[]) };
            let (labels, idx) = template::labels_with_mapping(&self.template, cands);
            let mut all: Vec<SpanPrediction> = Vec::new();
            for (c, chunk) in labels.chunks(self.max_labels.max(1)).enumerate() {
                let mut v = self
                    .model
                    .predict(SpanInput { texts: vec![text.clone()], labels: chunk.to_vec(), label_embeddings: None, input_spans: vec![spans[i].clone()] })?
                    .remove(0);
                for p in &mut v {
                    if let Some(j) = chunk.iter().position(|l| *l == p.label) {
                        p.entity_id = Some(cands[idx[c * self.max_labels.max(1) + j]].entity_id.clone());
                    }
                }
                all.extend(v);
            }
            // deduplicate_entities: keep the best per (text, start, end), first-seen order
            let mut best: Vec<SpanPrediction> = Vec::new();
            for p in all {
                match best.iter_mut().find(|b| b.start == p.start && b.end == p.end && b.text == p.text) {
                    Some(b) if p.score > b.score => *b = p,
                    Some(_) => {}
                    None => best.push(p),
                }
            }
            best.retain(|p| p.score >= self.model.params.threshold);
            best.sort_by_key(|p| p.start);
            out.push(best);
        }
        Ok(out)
    }
}
