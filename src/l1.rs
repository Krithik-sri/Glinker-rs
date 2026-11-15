//! L1 — mention detection.
//!
//! * [`L1::Gliner`]: zero-shot NER with a GLiNER span model (Python `L1GlinerComponent`),
//!   uni- or bi-encoder; bi-encoder label embeddings are computed once at load.
//! * [`L1::Gazetteer`]: dictionary matcher over entity labels/aliases (longest match, word
//!   boundaries). A model-free stand-in for the spaCy L1, which has no Rust equivalent.
//! * [`L1::Provided`]: mentions supplied by the caller (Python simple pipeline with spans).

use ndarray::Array2;
use crate::gliner::engine::{LabelEncoder, SpanModel};
use crate::gliner::pipeline::SpanInput;
use crate::text::{context, WordSplitter};
use crate::types::{DatabaseRecord, Mention};
use crate::Result;

#[derive(Clone, Debug)]
pub struct L1Options {
    pub max_left_context: usize,
    pub max_right_context: usize,
    pub min_entity_length: usize,
}
impl Default for L1Options {
    fn default() -> Self {
        Self { max_left_context: 50, max_right_context: 50, min_entity_length: 2 }
    }
}

pub enum L1 {
    Gliner { model: SpanModel, labels: Vec<String>, embeddings: Option<Array2<f32>>, opts: L1Options },
    Gazetteer { terms: Vec<(Vec<String>, String)>, splitter: WordSplitter, opts: L1Options },
    Provided { opts: L1Options },
}

impl L1 {
    pub fn gliner(model: SpanModel, labels: Vec<String>, encoder: Option<&LabelEncoder>, opts: L1Options) -> Result<Self> {
        let embeddings = if model.is_bi_encoder() {
            Some(encoder.ok_or("bi-encoder L1 needs a label encoder")?.encode(&labels)?)
        } else {
            None
        };
        Ok(L1::Gliner { model, labels, embeddings, opts })
    }

    /// Build a gazetteer from entity labels + aliases (lower-cased word sequences).
    pub fn gazetteer(entities: &[DatabaseRecord], opts: L1Options) -> Self {
        let splitter = WordSplitter::default();
        let mut terms: Vec<(Vec<String>, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for e in entities {
            for name in std::iter::once(&e.label).chain(e.aliases.iter()) {
                let toks: Vec<String> = splitter.split(name, usize::MAX).into_iter().map(|w| w.text.to_lowercase()).collect();
                if !toks.is_empty() && seen.insert(toks.clone()) {
                    terms.push((toks, e.entity_type.clone()));
                }
            }
        }
        terms.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // longest first
        L1::Gazetteer { terms, splitter, opts }
    }

    fn opts(&self) -> &L1Options {
        match self {
            L1::Gliner { opts, .. } | L1::Gazetteer { opts, .. } | L1::Provided { opts } => opts,
        }
    }

    pub fn extract(&self, texts: &[String]) -> Result<Vec<Vec<Mention>>> {
        let raw: Vec<Vec<(usize, usize, Option<String>)>> = match self {
            L1::Provided { .. } => return Err("L1::Provided: pass mentions to Executor::link_with_mentions".into()),
            L1::Gliner { model, labels, embeddings, .. } => {
                let preds = model.predict(SpanInput {
                    texts: texts.to_vec(),
                    labels: labels.clone(),
                    label_embeddings: embeddings.clone(),
                    input_spans: vec![None; texts.len()],
                })?;
                preds.into_iter().map(|v| v.into_iter().map(|p| (p.start, p.end, Some(p.label))).collect()).collect()
            }
            L1::Gazetteer { terms, splitter, .. } => texts
                .iter()
                .map(|t| {
                    let words = splitter.split(t, usize::MAX);
                    let lw: Vec<String> = words.iter().map(|w| w.text.to_lowercase()).collect();
                    let mut out = Vec::new();
                    let mut i = 0;
                    while i < words.len() {
                        let hit = terms.iter().find(|(toks, _)| i + toks.len() <= lw.len() && lw[i..i + toks.len()] == toks[..]);
                        if let Some((toks, ty)) = hit {
                            let j = i + toks.len() - 1;
                            out.push((words[i].start, words[j].end, (!ty.is_empty()).then(|| ty.clone())));
                            i = j + 1;
                        } else {
                            i += 1;
                        }
                    }
                    out
                })
                .collect(),
        };
        Ok(texts.iter().zip(raw).map(|(t, spans)| self.finish(t, spans)).collect())
    }

    /// Context + `filter_by_length` + dedupe + sort (Python L1 default pipeline).
    pub fn finish(&self, text: &str, spans: Vec<(usize, usize, Option<String>)>) -> Vec<Mention> {
        let o = self.opts();
        let mut seen = std::collections::HashSet::new();
        let mut v: Vec<Mention> = spans
            .into_iter()
            .filter(|(s, e, _)| seen.insert((*s, *e)))
            .map(|(s, e, label)| {
                let (l, r) = context(text, s, e, o.max_left_context, o.max_right_context);
                Mention { text: crate::text::char_slice(text, s, e), label, start: s, end: e, left_context: l, right_context: r }
            })
            .filter(|m| m.text.chars().count() >= o.min_entity_length)
            .collect();
        v.sort_by_key(|m| m.start);
        v
    }
}
