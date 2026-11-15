//! JSON pipeline config — same shape as GLinker's `ConfigBuilder` layers.
//!
//! ```json
//! {
//!   "name": "biomed",
//!   "template": "{label}: {description}",
//!   "entities": "data/entities.jsonl",
//!   "precompute_embeddings": true,
//!   "runtime": { "threads": 4 },
//!   "l1": { "type": "gliner", "model_dir": "models/ner", "labels": ["gene", "disease"], "threshold": 0.3 },
//!   "l2": { "max_candidates": 30, "search_mode": ["exact", "fuzzy"] },
//!   "l3": { "model_dir": "models/linker", "model_id": "knowledgator/gliner-linker-base-v1.0", "threshold": 0.5 },
//!   "l4": { "model_dir": "models/rerank", "max_labels": 20 },
//!   "l0": { "strict_matching": true, "position_tolerance": 2 }
//! }
//! ```
//! `"l1": {"type": "none"}` gives the Python *simple* pipeline (all entities are candidates,
//! loose L0 matching).

use serde::Deserialize;
use crate::l2::{FuzzyConfig, SearchMode};

#[derive(Clone, Debug, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default = "d_tpl")]
    pub template: String,
    #[serde(default)]
    pub entities: Option<String>,
    #[serde(default)]
    pub precompute_embeddings: bool,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    pub l1: L1Config,
    #[serde(default)]
    pub l2: L2Config,
    pub l3: ModelConfig,
    #[serde(default)]
    pub l4: Option<ModelConfig>,
    #[serde(default)]
    pub l0: L0Config,
}
fn d_tpl() -> String { "{label}".into() }

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "d_threads")]
    pub threads: usize,
}
fn d_threads() -> usize { 4 }
impl Default for RuntimeConfig {
    fn default() -> Self { Self { threads: 4 } }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum L1Config {
    None,
    Gazetteer {
        #[serde(default = "d_min_len")]
        min_entity_length: usize,
    },
    Gliner {
        model_dir: String,
        labels: Vec<String>,
        #[serde(default = "d_l1_thr")]
        threshold: f32,
        #[serde(default = "d_min_len")]
        min_entity_length: usize,
        #[serde(default = "d_label_tokens")]
        label_max_tokens: usize,
    },
}
fn d_min_len() -> usize { 2 }
fn d_l1_thr() -> f32 { 0.3 }

#[derive(Clone, Debug, Deserialize)]
pub struct L2Config {
    /// `"search"` (default): per-mention retrieval. `"all"`: whole database is every text's
    /// candidate set (Python `mentions=None`), while L1 spans still constrain L3.
    #[serde(default = "d_l2_mode")]
    pub mode: String,
    #[serde(default = "d_max_c")]
    pub max_candidates: usize,
    #[serde(default)]
    pub min_popularity: i64,
    #[serde(default = "d_modes")]
    pub search_mode: Vec<SearchMode>,
    #[serde(default)]
    pub fuzzy: FuzzyConfig,
}
fn d_max_c() -> usize { 30 }
fn d_l2_mode() -> String { "search".into() }
fn d_modes() -> Vec<SearchMode> { vec![SearchMode::Exact] }
impl Default for L2Config {
    fn default() -> Self { Self { mode: d_l2_mode(), max_candidates: 30, min_popularity: 0, search_mode: d_modes(), fuzzy: FuzzyConfig::default() } }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub model_dir: String,
    #[serde(default)]
    pub model_id: String,
    #[serde(default = "d_thr")]
    pub threshold: f32,
    #[serde(default = "d_true")]
    pub flat_ner: bool,
    #[serde(default)]
    pub multi_label: bool,
    #[serde(default = "d_true")]
    pub use_precomputed_embeddings: bool,
    #[serde(default)]
    pub cache_embeddings: bool,
    #[serde(default = "d_label_tokens")]
    pub label_max_tokens: usize,
    /// L4 only
    #[serde(default = "d_max_labels")]
    pub max_labels: usize,
}
fn d_thr() -> f32 { 0.5 }
fn d_true() -> bool { true }
fn d_label_tokens() -> usize { 128 }
fn d_max_labels() -> usize { 20 }

#[derive(Clone, Debug, Deserialize)]
pub struct L0Config {
    #[serde(default)]
    pub min_confidence: f32,
    #[serde(default = "d_true")]
    pub include_unlinked: bool,
    #[serde(default)]
    pub strict_matching: Option<bool>,
    #[serde(default = "d_tol")]
    pub position_tolerance: usize,
}
fn d_tol() -> usize { 2 }
impl Default for L0Config {
    fn default() -> Self { Self { min_confidence: 0.0, include_unlinked: true, strict_matching: None, position_tolerance: 2 } }
}
