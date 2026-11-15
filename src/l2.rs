//! L2 — candidate retrieval (Python `DatabaseChainComponent` + `DictLayer`).
//!
//! Layers are tried by descending priority; the first layer returning anything wins.
//! Per-mention pipeline: search → filter_by_popularity → dedupe → limit → sort_by_popularity.
//! Redis/Elasticsearch/Postgres layers plug in through [`CandidateLayer`].

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use serde::Deserialize;
use crate::fuzz;
use crate::types::DatabaseRecord;
use crate::Result;

#[derive(Clone, Debug, Deserialize)]
pub struct FuzzyConfig {
    #[serde(default = "d_min_sim")]
    pub min_similarity: f64,
    #[serde(default = "d_prefix")]
    pub prefix_length: usize,
}
fn d_min_sim() -> f64 { 0.3 }
fn d_prefix() -> usize { 1 }
impl Default for FuzzyConfig {
    fn default() -> Self { Self { min_similarity: 0.3, prefix_length: 1 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode { Exact, Fuzzy }

pub trait CandidateLayer: Send + Sync {
    fn priority(&self) -> i32;
    fn search_modes(&self) -> &[SearchMode];
    fn search(&self, query: &str) -> Vec<DatabaseRecord>;
    fn search_fuzzy(&self, _query: &str) -> Vec<DatabaseRecord> { vec![] }
    fn all(&self) -> Vec<DatabaseRecord>;
    fn update_embeddings(&mut self, ids: &[String], emb: &[Vec<f32>], model_id: &str) -> usize;
}

/// In-memory layer with label/alias indexes (Python `DictLayer`).
pub struct DictLayer {
    pub priority: i32,
    pub modes: Vec<SearchMode>,
    pub fuzzy: FuzzyConfig,
    storage: Vec<DatabaseRecord>,
    by_id: HashMap<String, usize>,
    label_index: HashMap<String, usize>,
    alias_index: HashMap<String, Vec<usize>>,
}

impl DictLayer {
    pub fn new(priority: i32, modes: Vec<SearchMode>, fuzzy: FuzzyConfig) -> Self {
        Self { priority, modes, fuzzy, storage: vec![], by_id: HashMap::new(), label_index: HashMap::new(), alias_index: HashMap::new() }
    }

    pub fn load_bulk(&mut self, entities: impl IntoIterator<Item = DatabaseRecord>, overwrite: bool) -> usize {
        let mut n = 0;
        for mut e in entities {
            if e.source.is_empty() { e.source = "dict".into(); }
            let slot = match self.by_id.get(&e.entity_id) {
                Some(_) if !overwrite => continue,
                Some(&i) => { self.storage[i] = e.clone(); i }
                None => { self.storage.push(e.clone()); self.storage.len() - 1 }
            };
            self.by_id.insert(e.entity_id.clone(), slot);
            self.label_index.insert(e.label.to_lowercase(), slot);
            for a in &e.aliases {
                let v = self.alias_index.entry(a.to_lowercase()).or_default();
                if !v.contains(&slot) { v.push(slot); }
            }
            n += 1;
        }
        n
    }

    /// Load `entities.jsonl` (one DatabaseRecord JSON per line) or a JSON array.
    pub fn load_file(&mut self, path: &str) -> Result<usize> {
        let text = std::fs::read_to_string(path)?;
        let recs: Vec<DatabaseRecord> = if text.trim_start().starts_with('[') {
            serde_json::from_str(&text)?
        } else {
            let mut v = Vec::new();
            for line in std::io::Cursor::new(text).lines() {
                let line = line?;
                if !line.trim().is_empty() { v.push(serde_json::from_str(&line)?); }
            }
            v
        };
        Ok(self.load_bulk(recs, true))
    }

    pub fn len(&self) -> usize { self.storage.len() }
    pub fn is_empty(&self) -> bool { self.storage.is_empty() }
}

impl CandidateLayer for DictLayer {
    fn priority(&self) -> i32 { self.priority }
    fn search_modes(&self) -> &[SearchMode] { &self.modes }

    fn search(&self, query: &str) -> Vec<DatabaseRecord> {
        let q = query.trim().to_lowercase();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        if let Some(&i) = self.label_index.get(&q) {
            seen.insert(i);
            out.push(self.storage[i].clone());
        }
        if let Some(v) = self.alias_index.get(&q) {
            for &i in v {
                if seen.insert(i) { out.push(self.storage[i].clone()); }
            }
        }
        out
    }

    fn search_fuzzy(&self, query: &str) -> Vec<DatabaseRecord> {
        let q = query.trim().to_lowercase();
        let prefix: String = q.chars().take(self.fuzzy.prefix_length).collect();
        let ok_prefix = |s: &str| self.fuzzy.prefix_length == 0 || s.starts_with(&prefix);
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (i, e) in self.storage.iter().enumerate() {
            let l = e.label.to_lowercase();
            if !ok_prefix(&l) { continue; } // Python skips aliases too when the label prefix fails
            let s = fuzz::ratio(&q, &l);
            if s >= self.fuzzy.min_similarity {
                scored.push((s, i));
                continue;
            }
            for a in &e.aliases {
                let a = a.to_lowercase();
                if !ok_prefix(&a) { continue; }
                let s = fuzz::ratio(&q, &a);
                if s >= self.fuzzy.min_similarity { scored.push((s, i)); break; }
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, i)| self.storage[i].clone()).collect()
    }

    fn all(&self) -> Vec<DatabaseRecord> { self.storage.clone() }

    fn update_embeddings(&mut self, ids: &[String], emb: &[Vec<f32>], model_id: &str) -> usize {
        let mut n = 0;
        for (id, e) in ids.iter().zip(emb) {
            if let Some(&i) = self.by_id.get(id) {
                self.storage[i].embedding = Some(e.clone());
                self.storage[i].embedding_model_id = Some(model_id.to_string());
                n += 1;
            }
        }
        // keep indexes pointing at the same slots; records were updated in place
        n
    }
}

#[derive(Clone, Debug)]
pub struct L2Options {
    pub max_candidates: usize,
    pub min_popularity: i64,
}
impl Default for L2Options {
    fn default() -> Self { Self { max_candidates: 30, min_popularity: 0 } }
}

pub struct L2 {
    pub layers: Vec<Box<dyn CandidateLayer>>,
    pub opts: L2Options,
}

impl L2 {
    pub fn new(mut layers: Vec<Box<dyn CandidateLayer>>, opts: L2Options) -> Self {
        layers.sort_by(|a, b| b.priority().cmp(&a.priority()));
        Self { layers, opts }
    }

    fn dedupe(v: Vec<DatabaseRecord>) -> Vec<DatabaseRecord> {
        let mut seen = HashSet::new();
        v.into_iter().filter(|r| seen.insert(r.entity_id.clone())).collect()
    }

    /// Candidates for one mention string.
    pub fn search(&self, mention: &str) -> Vec<DatabaseRecord> {
        let mut found = Vec::new();
        for layer in &self.layers {
            let mut r = Vec::new();
            for m in layer.search_modes() {
                match m {
                    SearchMode::Exact => r.extend(layer.search(mention)),
                    SearchMode::Fuzzy => r.extend(layer.search_fuzzy(mention)),
                }
            }
            if !r.is_empty() { found = Self::dedupe(r); break; }
        }
        let mut v: Vec<_> = found.into_iter().filter(|r| r.popularity >= self.opts.min_popularity).collect();
        v = Self::dedupe(v);
        v.truncate(self.opts.max_candidates);
        v.sort_by(|a, b| b.popularity.cmp(&a.popularity)); // stable, like Python sorted
        v
    }

    /// Whole database (Python `mentions=None` → simple pipeline).
    pub fn all(&self) -> Vec<DatabaseRecord> {
        Self::dedupe(self.layers.iter().flat_map(|l| l.all()).collect())
    }

    /// Cache embeddings into the highest-priority layer (Python `_cache_embeddings`).
    pub fn update_embeddings(&mut self, ids: &[String], emb: &[Vec<f32>], model_id: &str) -> usize {
        self.layers.first_mut().map(|l| l.update_embeddings(ids, emb, model_id)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rec(id: &str, label: &str, aliases: &[&str], pop: i64) -> DatabaseRecord {
        DatabaseRecord { entity_id: id.into(), label: label.into(), aliases: aliases.iter().map(|s| s.to_string()).collect(), popularity: pop, ..Default::default() }
    }
    #[test]
    fn exact_alias_fuzzy() {
        let mut d = DictLayer::new(0, vec![SearchMode::Exact, SearchMode::Fuzzy], FuzzyConfig { min_similarity: 0.8, prefix_length: 1 });
        d.load_bulk([rec("Q1", "insulin", &["INS"], 5), rec("Q2", "insulin receptor", &[], 9), rec("Q3", "glucagon", &[], 1)], false);
        let l2 = L2::new(vec![Box::new(d)], L2Options::default());
        let r = l2.search("Insulin");
        assert_eq!(r[0].entity_id, "Q1"); // exact first, fuzzy "insuline"-like extras filtered by 0.8
        assert_eq!(l2.search("ins")[0].entity_id, "Q1");
        assert_eq!(l2.search("insuline")[0].entity_id, "Q1");
        assert!(l2.search("zzz").is_empty());
    }
}
