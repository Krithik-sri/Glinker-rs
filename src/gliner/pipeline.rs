//! `orp` pipeline for GLiNER span models.
//!
//! pre:  SpanInput ─Split─▶ Prepared ─Encode─▶ (SessionInputs, SpanContext)
//! post: (SessionOutputs, SpanContext) ─Decode─▶ Vec<Vec<SpanPrediction>>

use std::collections::HashSet;
use composable::{composed, Composable};
use ndarray::Array2;
use orp::pipeline::{Pipeline, PostProcessor, PreProcessor};
use ort::session::{SessionInputs, SessionOutputs};
use tokenizers::Tokenizer;

use super::decode::{decode_item, greedy};
use super::encode::{encode, span_tensors, uni_prompt};
use super::{GlinerParams, Pooling, PromptKind};
use crate::text::{Word, WordSplitter};
use crate::types::SpanPrediction;
use crate::Result;

/// Request for one forward pass over a batch of texts sharing one label set.
pub struct SpanInput {
    pub texts: Vec<String>,
    pub labels: Vec<String>,
    /// Required for bi-encoders: (num_labels, hidden) in `labels` order.
    pub label_embeddings: Option<Array2<f32>>,
    /// Per text: `None` = score all spans; `Some(spans)` = only these char spans (GLinker `input_spans`).
    pub input_spans: Vec<Option<Vec<(usize, usize)>>>,
}

pub struct SpanContext {
    pub texts: Vec<String>,
    pub words: Vec<Vec<Word>>,
    pub labels: Vec<String>,
    pub filters: Vec<Option<HashSet<(usize, usize)>>>,
}

struct Prepared {
    input: SpanInput,
    words: Vec<Vec<Word>>,
    filters: Vec<Option<HashSet<(usize, usize)>>>,
}

pub const BASE_INPUTS: [&str; 6] = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"];
pub const LABELS_EMBEDDINGS: &str = "labels_embeddings";
pub const LOGITS: &str = "logits";

pub struct SpanPipeline {
    tokenizer: Tokenizer,
    splitter: WordSplitter,
    kind: PromptKind,
    pooling: Pooling,
}

impl SpanPipeline {
    pub fn new(tokenizer: Tokenizer, kind: PromptKind, pooling: Pooling) -> Self {
        Self { tokenizer, splitter: WordSplitter::default(), kind, pooling }
    }
    pub fn kind(&self) -> &PromptKind {
        &self.kind
    }
    pub fn splitter(&self) -> &WordSplitter {
        &self.splitter
    }
}

/// Char spans → word spans (Python `_convert_spans_to_word_indices`; misaligned spans dropped).
pub fn char_to_word_spans(words: &[Word], spans: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    let starts: std::collections::HashMap<usize, usize> = words.iter().enumerate().map(|(i, w)| (w.start, i)).collect();
    let ends: std::collections::HashMap<usize, usize> = words.iter().enumerate().map(|(i, w)| (w.end, i)).collect();
    spans
        .iter()
        .filter_map(|(s, e)| match (starts.get(s), ends.get(e)) {
            (Some(&a), Some(&b)) if b >= a => Some((a, b)),
            _ => None,
        })
        .collect()
}

struct Split<'p> {
    splitter: &'p WordSplitter,
    max_words: usize,
}
impl Composable<SpanInput, Prepared> for Split<'_> {
    fn apply(&self, input: SpanInput) -> Result<Prepared> {
        if input.input_spans.len() != input.texts.len() {
            return Err("input_spans must have one entry per text".into());
        }
        let words: Vec<Vec<Word>> = input.texts.iter().map(|t| self.splitter.split(t, self.max_words)).collect();
        let filters = words
            .iter()
            .zip(&input.input_spans)
            .map(|(w, s)| s.as_ref().map(|s| char_to_word_spans(w, s)))
            .collect();
        Ok(Prepared { input, words, filters })
    }
}

struct Encode<'p> {
    tokenizer: &'p Tokenizer,
    kind: &'p PromptKind,
    pooling: Pooling,
    max_width: usize,
}
impl<'a> Composable<Prepared, (SessionInputs<'a, 'a>, SpanContext)> for Encode<'_> {
    fn apply(&self, p: Prepared) -> Result<(SessionInputs<'a, 'a>, SpanContext)> {
        let prompt = match self.kind {
            PromptKind::BiEncoder => vec![],
            PromptKind::UniEncoder { ent_token, sep_token } => uni_prompt(&p.input.labels, ent_token, sep_token),
        };
        let enc = encode(self.tokenizer, &prompt, &p.words, self.pooling)?;
        let lens: Vec<usize> = p.words.iter().map(Vec::len).collect();
        let (span_idx, span_mask) = span_tensors(&lens, enc.num_words, self.max_width);
        let inputs: SessionInputs<'a, 'a> = match self.kind {
            PromptKind::BiEncoder => {
                let emb = p.input.label_embeddings.ok_or("bi-encoder requires label_embeddings")?;
                if emb.nrows() != p.input.labels.len() {
                    return Err("label_embeddings rows != labels".into());
                }
                ort::inputs! {
                    "input_ids" => enc.input_ids,
                    "attention_mask" => enc.attention_mask,
                    "words_mask" => enc.words_mask,
                    "text_lengths" => enc.text_lengths,
                    "span_idx" => span_idx,
                    "span_mask" => span_mask,
                    LABELS_EMBEDDINGS => emb,
                }?
                .into()
            }
            PromptKind::UniEncoder { .. } => ort::inputs! {
                "input_ids" => enc.input_ids,
                "attention_mask" => enc.attention_mask,
                "words_mask" => enc.words_mask,
                "text_lengths" => enc.text_lengths,
                "span_idx" => span_idx,
                "span_mask" => span_mask,
            }?
            .into(),
        };
        let ctx = SpanContext { texts: p.input.texts, words: p.words, labels: p.input.labels, filters: p.filters };
        Ok((inputs, ctx))
    }
}

struct Decode {
    params: GlinerParams,
}
impl<'a> Composable<(SessionOutputs<'a, 'a>, SpanContext), Vec<Vec<SpanPrediction>>> for Decode {
    fn apply(&self, (out, ctx): (SessionOutputs<'a, 'a>, SpanContext)) -> Result<Vec<Vec<SpanPrediction>>> {
        let logits = out.get(LOGITS).ok_or("model output has no `logits`")?;
        let arr = logits.try_extract_tensor::<f32>()?;
        let shape = arr.shape().to_vec();
        if shape.len() != 4 || shape[0] != ctx.texts.len() || shape[3] != ctx.labels.len() {
            return Err(format!("unexpected logits shape {shape:?}").into());
        }
        let mut result = Vec::with_capacity(ctx.texts.len());
        for (b, words) in ctx.words.iter().enumerate() {
            let item = arr.slice(ndarray::s![b, .., .., ..]);
            let raw = decode_item(item, words.len(), ctx.filters[b].as_ref(), self.params.threshold, self.params.top_k_class_probs);
            let kept = greedy(raw, self.params.flat_ner, self.params.multi_label);
            let text = &ctx.texts[b];
            result.push(
                kept.into_iter()
                    .map(|s| SpanPrediction {
                        text: text[words[s.start].byte_start..words[s.end].byte_end].to_string(),
                        label: ctx.labels[s.class].clone(),
                        start: words[s.start].start,
                        end: words[s.end].end,
                        score: s.score,
                        class_probs: s.class_probs.iter().map(|(c, p)| (ctx.labels[*c].clone(), *p)).collect(),
                        entity_id: None,
                    })
                    .collect(),
            );
        }
        Ok(result)
    }
}

impl<'a> Pipeline<'a> for SpanPipeline {
    type Input = SpanInput;
    type Output = Vec<Vec<SpanPrediction>>;
    type Context = SpanContext;
    type Parameters = GlinerParams;

    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            Split { splitter: &self.splitter, max_words: params.max_words },
            Encode { tokenizer: &self.tokenizer, kind: &self.kind, pooling: self.pooling, max_width: params.max_width }
        ]
    }

    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        Decode { params: params.clone() }
    }

    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        let extra = matches!(self.kind, PromptKind::BiEncoder).then_some(LABELS_EMBEDDINGS);
        Some(BASE_INPUTS.into_iter().chain(extra))
    }

    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some([LOGITS].into_iter())
    }
}
