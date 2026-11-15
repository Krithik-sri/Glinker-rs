//! L3 — entity linking with a GLiNER (bi-)encoder (Python `L3Processor`/`L3Component`).
//!
//! Candidate labels are rendered with the schema template, deduped case-insensitively,
//! scored against L1 mention spans (`input_spans`), and each prediction carries the
//! `entity_id` of the candidate that produced its label (no string re-matching needed).
//! Bi-encoder label embeddings are taken from candidates when every one has an embedding
//! for this model (`use_precomputed_embeddings`), otherwise computed by the label encoder.

use ndarray::Array2;
use crate::gliner::engine::{LabelEncoder, SpanModel};
use crate::gliner::pipeline::SpanInput;
use crate::template;
use crate::types::{DatabaseRecord, Mention, SpanPrediction};
use crate::Result;

pub struct L3 {
    pub model: SpanModel,
    pub encoder: Option<LabelEncoder>,
    pub template: String,
    pub use_precomputed_embeddings: bool,
}

/// Embeddings computed during a call, for write-back into L2 (`cache_embeddings`).
pub type NewEmbeddings = Vec<(String, Vec<f32>)>;

pub fn input_spans(mentions: Option<&[Vec<Mention>]>, n: usize) -> Vec<Option<Vec<(usize, usize)>>> {
    (0..n)
        .map(|i| {
            mentions
                .and_then(|m| m.get(i))
                .filter(|v| !v.is_empty()) // Python: empty L1 list → no constraint
                .map(|v| v.iter().map(|m| (m.start, m.end)).collect())
        })
        .collect()
}

impl L3 {
    fn embeddings_for(&self, cands: &[DatabaseRecord], idx: &[usize], labels: &[String], fresh: &mut NewEmbeddings) -> Result<Option<Array2<f32>>> {
        if !self.model.is_bi_encoder() {
            return Ok(None);
        }
        let picked: Vec<&DatabaseRecord> = idx.iter().map(|&i| &cands[i]).collect();
        let pre = self.use_precomputed_embeddings
            && picked.iter().all(|c| c.embedding.is_some() && c.embedding_model_id.as_deref() == Some(self.model.model_id.as_str()));
        if pre && !picked.is_empty() {
            let h = picked[0].embedding.as_ref().unwrap().len();
            let mut a = Array2::<f32>::zeros((picked.len(), h));
            for (r, c) in picked.iter().enumerate() {
                let e = c.embedding.as_ref().unwrap();
                if e.len() != h {
                    return Err(format!("embedding dim mismatch for {}", c.entity_id).into());
                }
                a.row_mut(r).assign(&ndarray::ArrayView1::from(e.as_slice()));
            }
            return Ok(Some(a));
        }
        let enc = self.encoder.as_ref().ok_or("bi-encoder L3 needs a label encoder")?;
        let a = enc.encode(labels)?;
        for (r, c) in picked.iter().enumerate() {
            if c.embedding.is_none() {
                fresh.push((c.entity_id.clone(), a.row(r).to_vec()));
            }
        }
        Ok(Some(a))
    }

    fn attach_ids(preds: &mut [SpanPrediction], labels: &[String], idx: &[usize], cands: &[DatabaseRecord]) {
        for p in preds {
            if let Some(j) = labels.iter().position(|l| *l == p.label) {
                p.entity_id = Some(cands[idx[j]].entity_id.clone());
            }
        }
    }

    /// Link `texts` against per-text `candidates`. If `shared` is true all texts use
    /// `candidates[0]` and run in one batched forward pass.
    pub fn link(&self, texts: &[String], candidates: &[Vec<DatabaseRecord>], mentions: Option<&[Vec<Mention>]>, shared: bool) -> Result<(Vec<Vec<SpanPrediction>>, NewEmbeddings)> {
        let spans = input_spans(mentions, texts.len());
        let mut fresh = NewEmbeddings::new();
        if shared && !candidates.is_empty() {
            let cands = &candidates[0];
            let (labels, idx) = template::labels_with_mapping(&self.template, cands);
            let emb = self.embeddings_for(cands, &idx, &labels, &mut fresh)?;
            let mut out = self.model.predict(SpanInput { texts: texts.to_vec(), labels: labels.clone(), label_embeddings: emb, input_spans: spans })?;
            for v in &mut out {
                Self::attach_ids(v, &labels, &idx, cands);
            }
            return Ok((out, fresh));
        }
        let mut out = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let cands = candidates.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let (labels, idx) = template::labels_with_mapping(&self.template, cands);
            if labels.is_empty() {
                out.push(vec![]);
                continue;
            }
            let emb = self.embeddings_for(cands, &idx, &labels, &mut fresh)?;
            let mut v = self.model.predict(SpanInput { texts: vec![text.clone()], labels: labels.clone(), label_embeddings: emb, input_spans: vec![spans[i].clone()] })?.remove(0);
            Self::attach_ids(&mut v, &labels, &idx, cands);
            // Python L3 default pipeline tail: filter_by_score (already ≥ threshold) + sort_by_position
            v.sort_by_key(|p| p.start);
            out.push(v);
        }
        Ok((out, fresh))
    }

    /// Precompute embeddings for a whole entity set (Python `executor.precompute_embeddings()`).
    pub fn precompute(&self, entities: &[DatabaseRecord]) -> Result<NewEmbeddings> {
        let enc = match (&self.encoder, self.model.is_bi_encoder()) {
            (Some(e), true) => e,
            _ => return Ok(vec![]),
        };
        let labels: Vec<String> = entities.iter().map(|e| template::format(&self.template, e)).collect();
        let a = enc.encode(&labels)?;
        Ok(entities.iter().enumerate().map(|(i, e)| (e.entity_id.clone(), a.row(i).to_vec())).collect())
    }
}
