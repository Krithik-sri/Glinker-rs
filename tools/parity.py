"""Independent reference: GLiNER bi-encoder span inference (tokenize_inputs + prepare_word_mask +
prepare_span_idx + SpanDecoder with input_spans + greedy flat NER) in numpy/onnxruntime.
Compares against glinker-rs simple-mode output (all L3 spans surface as l3_only)."""
import json, re, subprocess, sys
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

mdir, cfg, binary = sys.argv[1], sys.argv[2], sys.argv[3]
texts = ["Insulin regulates glucose in patients with diabetes.", "TP53 mutations cause cancer.", "Glucose and insulin, the hormone."]
ents = [json.loads(l) for l in open(cfg.rsplit("/", 1)[0] + "/entities.jsonl")]
labels = [e["label"] for e in ents]

tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
ltok = Tokenizer.from_file(f"{mdir}/labels_tokenizer.json")
le = ort.InferenceSession(f"{mdir}/labels_encoder.onnx")
encs = ltok.encode_batch(labels)
T = max(len(e.ids) for e in encs)
ids = np.zeros((len(labels), T), np.int64); am = np.zeros_like(ids)
for i, e in enumerate(encs): ids[i, :len(e.ids)] = e.ids; am[i, :len(e.ids)] = 1
emb = le.run(["embeddings"], {"input_ids": ids, "attention_mask": am})[0]

splitter = re.compile(r"\w+(?:[-_]\w+)*|\S")
K = 12
sess = ort.InferenceSession(f"{mdir}/model.onnx")
ref = []
for text in texts:
    words = [(m.group(), m.start(), m.end()) for m in splitter.finditer(text)]
    e = tok.encode([w for w, _, _ in words], is_pretokenized=True)
    wm, prev = [], None
    for wid in e.word_ids:
        wm.append(0 if wid is None else (wid + 1 if wid != prev else 0)); prev = wid
    L = len(words)
    span_idx = np.array([[(i, i + k) if i + k < L else (0, 0) for k in range(K)] for i in range(L)], np.int64).reshape(1, L * K, 2)
    span_mask = np.array([[i + k < L for k in range(K)] for i in range(L)]).reshape(1, L * K)
    logits = sess.run(["logits"], {"input_ids": np.array([e.ids]), "attention_mask": np.ones((1, len(e.ids)), np.int64),
                                   "words_mask": np.array([wm]), "text_lengths": np.array([[L]]), "span_idx": span_idx,
                                   "span_mask": span_mask, "labels_embeddings": emb})[0][0]
    probs = 1 / (1 + np.exp(-logits))
    cand = [(s, s + k, c, float(probs[s, k, c])) for s, k, c in zip(*np.where(probs > 0.5)) if s + k + 1 <= L]
    cand.sort(key=lambda x: -x[3])
    kept = []
    for sp in cand:
        if not any(not (sp[0] > q[1] or q[0] > sp[1]) for q in kept): kept.append(sp)
    kept.sort(key=lambda x: x[0])
    ref.append(sorted((words[s][1], words[e_][2], labels[c], round(p, 4)) for s, e_, c, p in kept))

out = json.loads(subprocess.run([binary, "link", cfg, *texts], capture_output=True, text=True, check=True).stdout)
rs = [sorted((x["mention_start"], x["mention_end"], next(e["label"] for e in ents if e["entity_id"] == x["linked_entity"]["entity_id"]),
              round(x["linked_entity"]["confidence"], 4)) for x in t) for t in out["entities"]]
ok = True
for t, a, b in zip(texts, ref, rs):
    same = a == b
    ok &= same
    print("MATCH" if same else "DIFF ", t, "\n  ref:", a, "\n  rs: ", b)
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
