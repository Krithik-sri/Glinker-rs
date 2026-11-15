//! `orp` pipeline for the bi-encoder **label encoder** (Python `model.encode_labels`).
//!
//! Graph contract (written by the notebook's export step):
//!   inputs  `input_ids` i64 [N, T], `attention_mask` i64 [N, T]
//!   output  `embeddings` f32 [N, H]  (labels_encoder → labels_projection → masked mean-pool)
//! Mean pooling is masked, so padding to the longest label equals Python's `padding="max_length"`.

use composable::Composable;
use ndarray::Array2;
use orp::pipeline::{Pipeline, PostProcessor, PreProcessor};
use ort::session::{SessionInputs, SessionOutputs};
use tokenizers::Tokenizer;
use crate::Result;

pub const EMBEDDINGS: &str = "embeddings";

pub struct LabelEncoderPipeline {
    tokenizer: Tokenizer,
    /// Max label tokens (GLinker sets `labels_tokenizer.model_max_length = max_length`).
    pub max_tokens: usize,
}

impl LabelEncoderPipeline {
    pub fn new(tokenizer: Tokenizer, max_tokens: usize) -> Self {
        Self { tokenizer, max_tokens }
    }
}

struct Tokenize<'p> {
    tok: &'p Tokenizer,
    max_tokens: usize,
}
impl<'a> Composable<Vec<String>, (SessionInputs<'a, 'a>, usize)> for Tokenize<'_> {
    fn apply(&self, labels: Vec<String>) -> Result<(SessionInputs<'a, 'a>, usize)> {
        let encs = self.tok.encode_batch(labels.iter().map(String::as_str).collect::<Vec<_>>(), true)?;
        let t = encs.iter().map(|e| e.get_ids().len().min(self.max_tokens)).max().unwrap_or(1).max(1);
        let n = encs.len();
        let mut ids = Array2::<i64>::zeros((n, t));
        let mut mask = Array2::<i64>::zeros((n, t));
        for (i, e) in encs.iter().enumerate() {
            let mut row: Vec<u32> = e.get_ids().to_vec();
            if row.len() > self.max_tokens {
                // keep the final special token (e.g. [SEP]) like HF truncation
                let last = *row.last().unwrap();
                row.truncate(self.max_tokens);
                *row.last_mut().unwrap() = last;
            }
            for (j, id) in row.iter().enumerate() {
                ids[[i, j]] = *id as i64;
                mask[[i, j]] = 1;
            }
        }
        let inputs = ort::inputs! { "input_ids" => ids, "attention_mask" => mask }?.into();
        Ok((inputs, n))
    }
}

struct Extract;
impl<'a> Composable<(SessionOutputs<'a, 'a>, usize), Array2<f32>> for Extract {
    fn apply(&self, (out, n): (SessionOutputs<'a, 'a>, usize)) -> Result<Array2<f32>> {
        let v = out.get(EMBEDDINGS).ok_or("label encoder has no `embeddings` output")?;
        let a = v.try_extract_tensor::<f32>()?;
        let a = a.into_dimensionality::<ndarray::Ix2>()?.to_owned();
        if a.nrows() != n {
            return Err("label encoder returned wrong row count".into());
        }
        Ok(a)
    }
}

impl<'a> Pipeline<'a> for LabelEncoderPipeline {
    type Input = Vec<String>;
    type Output = Array2<f32>;
    type Context = usize;
    type Parameters = ();

    fn pre_processor(&self, _: &()) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        Tokenize { tok: &self.tokenizer, max_tokens: self.max_tokens }
    }
    fn post_processor(&self, _: &()) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        Extract
    }
    fn expected_inputs(&self, _: &()) -> Option<impl Iterator<Item = &str>> {
        Some(["input_ids", "attention_mask"].into_iter())
    }
    fn expected_outputs(&self, _: &()) -> Option<impl Iterator<Item = &str>> {
        Some([EMBEDDINGS].into_iter())
    }
}
