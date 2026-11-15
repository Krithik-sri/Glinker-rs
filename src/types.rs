//! Data model shared by all layers (mirrors `glinker.l*.models`).
//! All `start`/`end` offsets are **character** (Unicode scalar) offsets, like Python `str` indices.

use serde::{Deserialize, Serialize};

/// Unified entity record (Python `DatabaseRecord`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DatabaseRecord {
    pub entity_id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub entity_type: String,
    #[serde(default)]
    pub popularity: i64,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model_id: Option<String>,
}

/// L1 mention (Python `L1Entity`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mention {
    pub text: String,
    #[serde(default)]
    pub label: Option<String>,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub left_context: String,
    #[serde(default)]
    pub right_context: String,
}

/// L3/L4 span prediction (Python `L3Entity`), plus the resolved entity id.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpanPrediction {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
    /// Top-5 (label, probability), descending — Python `class_probs`.
    #[serde(default)]
    pub class_probs: Vec<(String, f32)>,
    /// Entity id of the candidate whose formatted label produced `label` (set by L3/L4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LinkedEntity {
    pub entity_id: String,
    pub label: String,
    pub confidence: f32,
    pub start: usize,
    pub end: usize,
    pub matched_text: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    L1Only,
    L2Found,
    L3Linked,
    L3Only,
}

/// Aggregated output (Python `L0Entity`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct L0Entity {
    pub mention_text: String,
    pub label: Option<String>,
    pub mention_start: usize,
    pub mention_end: usize,
    pub left_context: String,
    pub right_context: String,
    pub candidates: Vec<DatabaseRecord>,
    pub num_candidates: usize,
    pub linked_entity: Option<LinkedEntity>,
    pub is_linked: bool,
    /// entity_id -> probability
    pub candidate_scores: Vec<(String, f32)>,
    pub pipeline_stage: Stage,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Stats {
    pub total_mentions: usize,
    pub linked: usize,
    pub unlinked: usize,
    pub linking_rate: f64,
    pub l1_only: usize,
    pub l2_found: usize,
    pub l3_linked: usize,
    pub l3_only: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct L0Output {
    pub entities: Vec<Vec<L0Entity>>,
    pub stats: Stats,
}
