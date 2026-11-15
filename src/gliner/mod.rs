//! GLiNER span-mode inference on `orp` (uni-encoder and bi-encoder), with GLinker's
//! `input_spans` constraint and `class_probs` output — the two features `gline-rs` lacks
//! and GLinker's L3/L4 depend on.

pub mod decode;
pub mod encode;
pub mod engine;
pub mod labels;
pub mod pipeline;

use serde::Deserialize;

/// Inference parameters (Python `predict_entities(..., threshold, flat_ner, multi_label)`).
#[derive(Clone, Debug)]
pub struct GlinerParams {
    pub threshold: f32,
    pub flat_ner: bool,
    pub multi_label: bool,
    /// Maximum span width in words (`gliner_config.max_width`).
    pub max_width: usize,
    /// Maximum words per text (`gliner_config.max_len`).
    pub max_words: usize,
    /// Number of `class_probs` kept per span (Python hard-codes 5).
    pub top_k_class_probs: usize,
}

impl Default for GlinerParams {
    fn default() -> Self {
        Self { threshold: 0.5, flat_ner: true, multi_label: false, max_width: 12, max_words: 384, top_k_class_probs: 5 }
    }
}

/// How labels reach the model.
#[derive(Clone, Debug, PartialEq)]
pub enum PromptKind {
    /// Labels encoded separately; text graph takes `labels_embeddings` [C, H].
    BiEncoder,
    /// Labels are prepended to the text: `<<ENT>> l1 <<ENT>> l2 ... <<SEP>> words...`.
    UniEncoder { ent_token: String, sep_token: String },
}

/// Subset of `gliner_config.json` needed at inference time.
#[derive(Clone, Debug, Deserialize)]
pub struct GlinerConfigFile {
    #[serde(default = "d_max_width")]
    pub max_width: usize,
    #[serde(default = "d_max_len")]
    pub max_len: usize,
    #[serde(default = "d_ent")]
    pub ent_token: String,
    #[serde(default = "d_sep")]
    pub sep_token: String,
    #[serde(default)]
    pub labels_encoder: Option<String>,
    #[serde(default)]
    pub span_mode: Option<String>,
    #[serde(default = "d_pool")]
    pub subtoken_pooling: String,
    #[serde(default = "d_split")]
    pub words_splitter_type: String,
}
fn d_pool() -> String { "first".into() }
fn d_split() -> String { "whitespace".into() }

/// GLiNER `subtoken_pooling`: which sub-tokens get a word id in `words_mask`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pooling { First, Last, All }
impl Pooling {
    pub fn parse(s: &str) -> crate::Result<Self> {
        match s {
            "first" => Ok(Pooling::First),
            "last" => Ok(Pooling::Last),
            "mean" | "max" => Ok(Pooling::All),
            o => Err(format!("unsupported subtoken_pooling {o:?}").into()),
        }
    }
}
fn d_max_width() -> usize { 12 }
fn d_max_len() -> usize { 384 }
fn d_ent() -> String { "<<ENT>>".into() }
fn d_sep() -> String { "<<SEP>>".into() }

impl GlinerConfigFile {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> crate::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
    }
    pub fn pooling(&self) -> crate::Result<Pooling> {
        if self.words_splitter_type != "whitespace" {
            return Err(format!("words_splitter_type {:?} not ported (only the default whitespace regex)", self.words_splitter_type).into());
        }
        Pooling::parse(&self.subtoken_pooling)
    }
    pub fn prompt_kind(&self) -> PromptKind {
        if self.labels_encoder.is_some() {
            PromptKind::BiEncoder
        } else {
            PromptKind::UniEncoder { ent_token: self.ent_token.clone(), sep_token: self.sep_token.clone() }
        }
    }
}
