//! Sub-word encoding — port of GLiNER `tokenize_inputs` + `prepare_word_mask` (first-subtoken
//! pooling) + `prepare_span_idx`. Special tokens come from `tokenizer.json`'s post-processor,
//! so DeBERTa (`[CLS]=1`) and ModernBERT/Ettin (`[CLS]=50281`) backbones both work.

use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;
use crate::text::Word;
use super::Pooling;
use crate::Result;

pub struct Encoded {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub words_mask: Array2<i64>,
    pub text_lengths: Array2<i64>,
    /// max words over the batch (the `L` dimension of logits)
    pub num_words: usize,
}

/// Load a tokenizer and strip any truncation/padding baked into `tokenizer.json`.
pub fn load_tokenizer<P: AsRef<std::path::Path>>(p: P) -> Result<Tokenizer> {
    let mut t = Tokenizer::from_file(p)?;
    t.with_truncation(None)?;
    t.with_padding(None);
    Ok(t)
}

/// Encode `prompt ++ words` for each text. `prompt` is empty for bi-encoders.
pub fn encode(tokenizer: &Tokenizer, prompt: &[String], texts: &[Vec<Word>], pooling: Pooling) -> Result<Encoded> {
    let skip = prompt.len();
    let mut rows: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(texts.len());
    for words in texts {
        let seq: Vec<&str> = prompt.iter().map(String::as_str).chain(words.iter().map(|w| w.text.as_str())).collect();
        let enc = tokenizer.encode(seq.as_slice(), true)?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let wids = enc.get_word_ids();
        let mut mask = Vec::with_capacity(ids.len());
        let mut prev: Option<u32> = None;
        for (ti, wid) in wids.iter().enumerate() {
            match *wid {
                None => mask.push(0),
                Some(w) => {
                    let first = Some(w) != prev;
                    let last = wids.get(ti + 1).copied().flatten() != Some(w);
                    let pick = match pooling { Pooling::First => first, Pooling::Last => last, Pooling::All => true };
                    let w = w as usize;
                    mask.push(if w >= skip && pick { (w - skip + 1) as i64 } else { 0 });
                }
            }
            prev = *wid;
        }
        rows.push((ids, mask));
    }
    let b = texts.len();
    let t = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let mut input_ids = Array2::<i64>::zeros((b, t));
    let mut attention_mask = Array2::<i64>::zeros((b, t));
    let mut words_mask = Array2::<i64>::zeros((b, t));
    for (i, (ids, m)) in rows.iter().enumerate() {
        for j in 0..ids.len() {
            input_ids[[i, j]] = ids[j];
            attention_mask[[i, j]] = 1;
            words_mask[[i, j]] = m[j];
        }
    }
    let mut text_lengths = Array2::<i64>::zeros((b, 1));
    for (i, w) in texts.iter().enumerate() {
        text_lengths[[i, 0]] = w.len() as i64;
    }
    let num_words = texts.iter().map(Vec::len).max().unwrap_or(0);
    Ok(Encoded { input_ids, attention_mask, words_mask, text_lengths, num_words })
}

/// `prepare_span_idx` for a padded batch: shape (B, L·K, 2) and mask (B, L·K).
/// Span `(i, i+w)` is valid iff `i + w < len_b`. Lean: `span_count` proves the valid count.
pub fn span_tensors(lens: &[usize], num_words: usize, max_width: usize) -> (Array3<i64>, Array2<bool>) {
    let b = lens.len();
    let n = num_words * max_width;
    let mut idx = Array3::<i64>::zeros((b, n, 2));
    let mut mask = Array2::<bool>::from_elem((b, n), false);
    for (bi, &len) in lens.iter().enumerate() {
        for i in 0..num_words {
            for w in 0..max_width {
                let j = i * max_width + w;
                if i + w < len {
                    idx[[bi, j, 0]] = i as i64;
                    idx[[bi, j, 1]] = (i + w) as i64;
                    mask[[bi, j]] = true;
                }
            }
        }
    }
    (idx, mask)
}

/// GLiNER uni-encoder prompt: `[ent, label]* sep`.
pub fn uni_prompt(labels: &[String], ent_token: &str, sep_token: &str) -> Vec<String> {
    let mut p = Vec::with_capacity(labels.len() * 2 + 1);
    for l in labels {
        p.push(ent_token.to_string());
        if !l.is_empty() {
            p.push(l.clone());
        }
    }
    p.push(sep_token.to_string());
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn span_mask_count() {
        let (_, m) = span_tensors(&[4, 2], 4, 3);
        // len 4, K 3: 3+3+2+1 = 9 ; len 2: 2+1 = 3
        assert_eq!(m.row(0).iter().filter(|x| **x).count(), 9);
        assert_eq!(m.row(1).iter().filter(|x| **x).count(), 3);
    }
}
