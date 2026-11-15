//! DAG executor: L1 → L2 → L3 (or L4) → L0 (Python `DAGExecutor` for the built-in topologies).
//! When L4 is configured its output feeds L0, as in Python; L3 is then skipped because Python
//! discards its result.

use std::path::Path;
use orp::params::RuntimeParameters;

use crate::config::*;
use crate::gliner::engine::{LabelEncoder, SpanModel};
use crate::gliner::GlinerParams;
use crate::l0::{self, L0Options};
use crate::l1::{L1Options, L1};
use crate::l2::{DictLayer, L2Options, L2};
use crate::l3::L3;
use crate::l4::L4;
use crate::types::*;
use crate::Result;

pub struct Executor {
    pub name: String,
    pub l1: Option<L1>,
    pub l2: L2,
    pub l3: L3,
    pub l4: Option<L4>,
    pub l0: L0Options,
    pub template: String,
    pub cache_embeddings: bool,
    /// L2 `"mode": "all"`
    pub all_candidates: bool,
}

fn rt(c: &RuntimeConfig) -> RuntimeParameters {
    let p = RuntimeParameters::default().with_threads(c.threads);
    #[cfg(feature = "cuda")]
    let p = p.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()]);
    p
}

fn params(m: &ModelConfig) -> GlinerParams {
    GlinerParams { threshold: m.threshold, flat_ner: m.flat_ner, multi_label: m.multi_label, ..Default::default() }
}

impl Executor {
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let cfg: PipelineConfig = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let base = path.parent().unwrap_or(Path::new("."));
        Self::from_config(cfg, base)
    }

    /// Relative paths in `cfg` are resolved against `base`.
    pub fn from_config(cfg: PipelineConfig, base: &Path) -> Result<Self> {
        let res = |p: &str| base.join(p);
        let mut dict = DictLayer::new(0, cfg.l2.search_mode.clone(), cfg.l2.fuzzy.clone());
        if let Some(e) = &cfg.entities {
            dict.load_file(res(e).to_str().ok_or("bad path")?)?;
        }
        let entities_snapshot = {
            use crate::l2::CandidateLayer;
            dict.all()
        };
        let l2 = L2::new(vec![Box::new(dict)], L2Options { max_candidates: cfg.l2.max_candidates, min_popularity: cfg.l2.min_popularity });

        let l1 = match &cfg.l1 {
            L1Config::None => None,
            L1Config::Gazetteer { min_entity_length } => {
                Some(L1::gazetteer(&entities_snapshot, L1Options { min_entity_length: *min_entity_length, ..Default::default() }))
            }
            L1Config::Gliner { model_dir, labels, threshold, min_entity_length, label_max_tokens } => {
                let p = GlinerParams { threshold: *threshold, ..Default::default() };
                let m = SpanModel::load(res(model_dir), model_dir, p, rt(&cfg.runtime))?;
                let enc = if m.is_bi_encoder() { Some(LabelEncoder::load(res(model_dir), *label_max_tokens, rt(&cfg.runtime))?) } else { None };
                Some(L1::gliner(m, labels.clone(), enc.as_ref(), L1Options { min_entity_length: *min_entity_length, ..Default::default() })?)
            }
        };

        let m3 = &cfg.l3;
        let model_id = if m3.model_id.is_empty() { m3.model_dir.clone() } else { m3.model_id.clone() };
        let span = SpanModel::load(res(&m3.model_dir), &model_id, params(m3), rt(&cfg.runtime))?;
        let encoder = if span.is_bi_encoder() { Some(LabelEncoder::load(res(&m3.model_dir), m3.label_max_tokens, rt(&cfg.runtime))?) } else { None };
        let l3 = L3 { model: span, encoder, template: cfg.template.clone(), use_precomputed_embeddings: m3.use_precomputed_embeddings };

        let l4 = match &cfg.l4 {
            Some(m4) => {
                let id = if m4.model_id.is_empty() { m4.model_dir.clone() } else { m4.model_id.clone() };
                let model = SpanModel::load(res(&m4.model_dir), &id, params(m4), rt(&cfg.runtime))?;
                if model.is_bi_encoder() {
                    return Err("L4 reranker must be a uni-encoder GLiNER".into());
                }
                Some(L4 { model, template: cfg.template.clone(), max_labels: m4.max_labels })
            }
            None => None,
        };

        let simple = l1.is_none();
        let l0 = L0Options {
            min_confidence: cfg.l0.min_confidence,
            include_unlinked: cfg.l0.include_unlinked,
            strict_matching: cfg.l0.strict_matching.unwrap_or(!simple),
            position_tolerance: cfg.l0.position_tolerance,
        };
        if cfg.l2.mode != "search" && cfg.l2.mode != "all" {
            return Err(format!("unknown l2.mode {:?}", cfg.l2.mode).into());
        }
        let all_candidates = cfg.l2.mode == "all";
        let mut ex = Executor { name: cfg.name, l1, l2, l3, l4, l0, template: cfg.template, cache_embeddings: m3.cache_embeddings, all_candidates };
        if cfg.precompute_embeddings {
            ex.precompute_embeddings()?;
        }
        Ok(ex)
    }

    pub fn load_entities(&mut self, entities: Vec<DatabaseRecord>) -> Result<()> {
        let mut d = DictLayer::new(0, vec![crate::l2::SearchMode::Exact], Default::default());
        d.load_bulk(entities, true);
        self.l2.layers.push(Box::new(d));
        Ok(())
    }

    pub fn precompute_embeddings(&mut self) -> Result<usize> {
        let all = self.l2.all();
        let new = self.l3.precompute(&all)?;
        let (ids, embs): (Vec<_>, Vec<_>) = new.into_iter().unzip();
        let id = self.l3.model.model_id.clone();
        Ok(self.l2.update_embeddings(&ids, &embs, &id))
    }

    /// Full pipeline on raw texts.
    pub fn execute(&mut self, texts: &[String]) -> Result<L0Output> {
        let mentions = match &self.l1 {
            Some(l1) => Some(l1.extract(texts)?),
            None => None,
        };
        self.run(texts, mentions)
    }

    /// Pipeline with caller-supplied mentions (skips L1).
    pub fn link_with_mentions(&mut self, texts: &[String], mentions: Vec<Vec<Mention>>) -> Result<L0Output> {
        self.run(texts, Some(mentions))
    }

    fn run(&mut self, texts: &[String], mentions: Option<Vec<Vec<Mention>>>) -> Result<L0Output> {
        // L2
        let (per_mention, per_text, shared): (Vec<Vec<Vec<DatabaseRecord>>>, Vec<Vec<DatabaseRecord>>, bool) = match &mentions {
            None => (vec![vec![]; texts.len()], vec![self.l2.all()], true),
            Some(ms) => {
                let pm: Vec<Vec<Vec<DatabaseRecord>>> = ms.iter().map(|v| v.iter().map(|m| self.l2.search(&m.text)).collect()).collect();
                if self.all_candidates {
                    (pm, vec![self.l2.all()], true)
                } else {
                    let pt = pm.iter().map(|v| v.iter().flatten().cloned().collect()).collect();
                    (pm, pt, false)
                }
            }
        };
        // L3 / L4
        let links = if let Some(l4) = &self.l4 {
            l4.rerank(texts, &per_text, mentions.as_deref(), shared)?
        } else {
            let (links, fresh) = self.l3.link(texts, &per_text, mentions.as_deref(), shared)?;
            if self.cache_embeddings && !fresh.is_empty() {
                let (ids, embs): (Vec<_>, Vec<_>) = fresh.into_iter().unzip();
                let id = self.l3.model.model_id.clone();
                self.l2.update_embeddings(&ids, &embs, &id);
            }
            links
        };
        // L0
        let empty: Vec<Mention> = vec![];
        let entities: Vec<Vec<L0Entity>> = (0..texts.len())
            .map(|i| {
                let ms = mentions.as_ref().map(|m| m[i].as_slice()).unwrap_or(&empty);
                let tc = if shared { &per_text[0] } else { &per_text[i] };
                l0::aggregate_text(ms, &per_mention[i], tc, &links[i], &self.template, &self.l0)
            })
            .collect();
        let stats = l0::stats(&entities);
        Ok(L0Output { entities, stats })
    }
}
