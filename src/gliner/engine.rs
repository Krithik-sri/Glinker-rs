//! Model handles: an `orp::model::Model` bound to its pipeline and parameters.
//!
//! Expected model directory (produced by the notebook export cell):
//! ```text
//! <dir>/gliner_config.json
//! <dir>/tokenizer.json            text tokenizer
//! <dir>/model.onnx                span graph (bi-encoder: takes `labels_embeddings`)
//! <dir>/labels_tokenizer.json     bi-encoder only
//! <dir>/labels_encoder.onnx       bi-encoder only
//! ```

use std::path::Path;
use ndarray::{concatenate, Array2, Axis};
use orp::model::Model;
use orp::params::RuntimeParameters;

use super::encode::load_tokenizer;
use super::labels::LabelEncoderPipeline;
use super::pipeline::{SpanInput, SpanPipeline};
use super::{GlinerConfigFile, GlinerParams, PromptKind};
use crate::types::SpanPrediction;
use crate::Result;

pub struct SpanModel {
    model: Model,
    pipeline: SpanPipeline,
    pub params: GlinerParams,
    pub model_id: String,
}

impl SpanModel {
    pub fn load(dir: impl AsRef<Path>, model_id: &str, mut params: GlinerParams, rt: RuntimeParameters) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = GlinerConfigFile::from_file(dir.join("gliner_config.json"))?;
        params.max_width = cfg.max_width;
        params.max_words = params.max_words.min(cfg.max_len);
        let pipeline = SpanPipeline::new(load_tokenizer(dir.join("tokenizer.json"))?, cfg.prompt_kind(), cfg.pooling()?);
        let model = Model::new(dir.join("model.onnx"), rt)?;
        Ok(Self { model, pipeline, params, model_id: model_id.to_string() })
    }

    pub fn is_bi_encoder(&self) -> bool {
        matches!(self.pipeline.kind(), PromptKind::BiEncoder)
    }

    pub fn predict(&self, input: SpanInput) -> Result<Vec<Vec<SpanPrediction>>> {
        if input.labels.is_empty() {
            return Ok(vec![vec![]; input.texts.len()]);
        }
        self.model.inference(input, &self.pipeline, &self.params)
    }
}

pub struct LabelEncoder {
    model: Model,
    pipeline: LabelEncoderPipeline,
    pub batch_size: usize,
}

impl LabelEncoder {
    pub fn load(dir: impl AsRef<Path>, max_tokens: usize, rt: RuntimeParameters) -> Result<Self> {
        let dir = dir.as_ref();
        let pipeline = LabelEncoderPipeline::new(load_tokenizer(dir.join("labels_tokenizer.json"))?, max_tokens);
        let model = Model::new(dir.join("labels_encoder.onnx"), rt)?;
        Ok(Self { model, pipeline, batch_size: 32 })
    }

    /// Python `encode_labels(labels, batch_size)` → (N, H).
    pub fn encode(&self, labels: &[String]) -> Result<Array2<f32>> {
        let mut parts = Vec::new();
        for chunk in labels.chunks(self.batch_size.max(1)) {
            parts.push(self.model.inference(chunk.to_vec(), &self.pipeline, &())?);
        }
        if parts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
        Ok(concatenate(Axis(0), &views)?)
    }
}
