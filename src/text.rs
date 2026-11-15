//! Word splitting and offset utilities.
//!
//! GLiNER's default `WhitespaceTokenSplitter` uses `\w+(?:[-_]\w+)*|\S`; Rust's `regex`
//! `\w` is Unicode-aware like Python 3 `re`, so Tamil/Arabic text splits identically.

use regex::Regex;

/// A word with byte and char offsets into the source text.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub start: usize,
    pub end: usize,
}

pub struct WordSplitter {
    re: Regex,
}

impl Default for WordSplitter {
    fn default() -> Self {
        Self { re: Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("static regex") }
    }
}

impl WordSplitter {
    pub fn split(&self, text: &str, max_words: usize) -> Vec<Word> {
        let b2c = ByteToChar::new(text);
        self.re
            .find_iter(text)
            .take(max_words)
            .map(|m| Word {
                text: m.as_str().to_string(),
                byte_start: m.start(),
                byte_end: m.end(),
                start: b2c.get(m.start()),
                end: b2c.get(m.end()),
            })
            .collect()
    }
}

/// O(1) byte-offset → char-offset lookup.
pub struct ByteToChar {
    map: Vec<usize>,
}

impl ByteToChar {
    pub fn new(text: &str) -> Self {
        let mut map = vec![0usize; text.len() + 1];
        let mut c = 0usize;
        for (b, ch) in text.char_indices() {
            for i in 0..ch.len_utf8() {
                map[b + i] = c;
            }
            c += 1;
        }
        map[text.len()] = c;
        Self { map }
    }
    pub fn get(&self, byte: usize) -> usize {
        self.map[byte]
    }
}

/// Substring by char offsets.
pub fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end.saturating_sub(start)).collect()
}

/// Python L1 `_get_context`: up to `max_left`/`max_right` chars around a mention, stripped.
pub fn context(text: &str, start: usize, end: usize, max_left: usize, max_right: usize) -> (String, String) {
    let n = text.chars().count();
    let l = char_slice(text, start.saturating_sub(max_left), start);
    let r = char_slice(text, end, (end + max_right).min(n));
    (l.trim().to_string(), r.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_unicode() {
        let s = WordSplitter::default();
        let w = s.split("TP53-mutant தமிழ் x.", 100);
        let t: Vec<_> = w.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(t[0], "TP53-mutant");
        assert_eq!(*t.last().unwrap(), ".");
        // char offsets
        let tamil = &w[1];
        assert_eq!(tamil.start, 12);
        assert_eq!(char_slice("TP53-mutant தமிழ் x.", tamil.start, tamil.end), tamil.text);
    }
}
