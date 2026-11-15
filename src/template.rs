//! Label templates (Python `schema.template`, e.g. `"{label}: {description}"`).

use crate::types::DatabaseRecord;

/// Format a candidate with a `str.format`-style template.
/// Supported fields: label, description, entity_id, entity_type, popularity, aliases.
/// Unknown fields fall back to the plain label (Python's `KeyError` branch).
pub fn format(template: &str, r: &DatabaseRecord) -> String {
    let mut out = String::with_capacity(template.len() + r.label.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push_str(&rest[open..]);
            return out;
        };
        let field = &after[..close];
        match field {
            "label" => out.push_str(&r.label),
            "description" => out.push_str(&r.description),
            "entity_id" => out.push_str(&r.entity_id),
            "entity_type" => out.push_str(&r.entity_type),
            "popularity" => out.push_str(&r.popularity.to_string()),
            "aliases" => out.push_str(&format!("{:?}", r.aliases)),
            _ => return r.label.clone(),
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// Python `_create_gliner_labels_with_mapping`: format, dedupe case-insensitively,
/// keep first occurrence. Returns (labels, index into `candidates` per label).
pub fn labels_with_mapping(template: &str, candidates: &[DatabaseRecord]) -> (Vec<String>, Vec<usize>) {
    let mut seen = std::collections::HashSet::new();
    let mut labels = Vec::new();
    let mut idx = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let l = format(template, c);
        if seen.insert(l.to_lowercase()) {
            labels.push(l);
            idx.push(i);
        }
    }
    (labels, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fmt() {
        let r = DatabaseRecord { label: "insulin".into(), description: "hormone".into(), ..Default::default() };
        assert_eq!(format("{label}: {description}", &r), "insulin: hormone");
        assert_eq!(format("{nope}", &r), "insulin");
    }
}
