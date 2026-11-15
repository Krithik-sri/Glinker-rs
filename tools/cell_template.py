# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║ glinker-rs — Knowledgator GLinker → Rust on fbilhaut/orp + ort   (single-cell Colab)      ║
# ║ Steps: deps → write crate → export GLiNER to ONNX → Lean4 proofs → cargo test/build →     ║
# ║        parity vs Python GLiNER → MCC eval (cross-checked with sklearn) → OpenAI server    ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝
LINKER_MODEL = "knowledgator/gliner-linker-base-v1.0"   # bi-encoder (L3); large: ...-large-v1.0
NER_MODEL    = None    # e.g. "knowledgator/gliner-bi-base-v2.0" → GLiNER L1; None → gazetteer L1
RERANK_MODEL = None    # e.g. "knowledgator/gliner-linker-rerank-v1.0" → uni-encoder L4
THRESHOLD    = 0.5
PORT         = 8000
WORK         = "/content/glinker_work"

import os, sys, json, subprocess, shutil, time, pathlib, textwrap, glob
def sh(cmd, check=True, env=None, quiet=False):
    print(f"$ {cmd[:160]}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, env={**os.environ, **(env or {})})
    if not quiet or r.returncode:
        out = (r.stdout + r.stderr).strip()
        if out: print(out[-4000:])
    if check and r.returncode: raise RuntimeError(f"failed: {cmd}")
    return r.stdout

# ── 0. dependencies ───────────────────────────────────────────────────────────────────────
sh(f"{sys.executable} -m pip install -q gliner==0.2.30 onnx onnxruntime==1.20.1 openai scikit-learn", quiet=True)
if not os.path.exists(os.path.expanduser("~/.cargo/bin/cargo")):
    sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", quiet=True)
if not os.path.exists(os.path.expanduser("~/.elan/bin/lean")):
    sh("curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.15.0", quiet=True)
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"]
import onnxruntime as _ort
ORT_DIR = os.path.join(os.path.dirname(_ort.__file__), "capi")
ORT_SO = sorted(glob.glob(os.path.join(ORT_DIR, "libonnxruntime.so*")))[-1]
ENV = {"ORT_LIB_LOCATION": ORT_DIR, "ORT_DYLIB_PATH": ORT_SO, "RUST_BACKTRACE": "0"}
os.environ.update(ENV)

# ── 1. write the crate (sources tested before packaging) ──────────────────────────────────
FILES = __FILES__
CRATE = os.path.join(WORK, "glinker-rs")
for rel, body in FILES.items():
    p = pathlib.Path(CRATE, rel); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body)
print(f"wrote {len(FILES)} files to {CRATE}")

# ── 2. export GLiNER → ONNX (custom exporter; GLiNER 0.2.30's from_labels_embeddings path ──
#      passes the flag into _build_dummy_batch and zips inputs under the wrong names)
import torch
from gliner import GLiNER

def load_gliner(model_id):
    m = GLiNER.from_pretrained(model_id).eval(); m.to("cpu")
    lt = getattr(m.data_processor, "labels_tokenizer", None)
    if lt is not None and lt.model_max_length > 100000:   # same fix GLinker's L3Component applies
        lt.model_max_length = 512
    return m

def export_gliner(model_id, out_dir):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    m = load_gliner(model_id)
    if (out / "model.onnx").exists(): return m
    core = m.model.eval()
    probe = ["person", "organization", "country"]
    batch = m._build_dummy_batch(labels=probe)
    base = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "words_mask": {0: "batch", 1: "seq"},
            "text_lengths": {0: "batch"}, "span_idx": {0: "batch", 1: "num_spans"}, "span_mask": {0: "batch", 1: "num_spans"},
            "logits": {0: "batch", 1: "words", 2: "width", 3: "classes"}}
    bi = getattr(m.config, "labels_encoder", None) is not None
    if bi:
        emb = m.encode_labels(probe).cpu()
        class TextGraph(torch.nn.Module):
            def __init__(s, c): super().__init__(); s.c = c
            def forward(s, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask, labels_embeddings):
                o = s.c(input_ids=input_ids, attention_mask=attention_mask, words_mask=words_mask, text_lengths=text_lengths,
                        span_idx=span_idx, span_mask=span_mask, labels_embeds=labels_embeddings)
                return o.logits if hasattr(o, "logits") else o[0]
        torch.onnx.export(TextGraph(core), tuple(batch[k] for k in base) + (emb,), str(out / "model.onnx"),
                          input_names=base + ["labels_embeddings"], output_names=["logits"],
                          dynamic_axes={**axes, "labels_embeddings": {0: "classes"}}, opset_version=19, dynamo=False)
        class LabelGraph(torch.nn.Module):
            def __init__(s, c): super().__init__(); s.c = c
            def forward(s, input_ids, attention_mask):
                return s.c.token_rep_layer.encode_labels(input_ids, attention_mask)
        lt = m.data_processor.labels_tokenizer(probe, return_tensors="pt", padding=True)
        torch.onnx.export(LabelGraph(core), (lt["input_ids"], lt["attention_mask"]), str(out / "labels_encoder.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["embeddings"],
                          dynamic_axes={"input_ids": {0: "n", 1: "t"}, "attention_mask": {0: "n", 1: "t"}, "embeddings": {0: "n"}},
                          opset_version=19, dynamo=False)
        tmp = out / "_labels_tok"; m.data_processor.labels_tokenizer.save_pretrained(tmp)
        shutil.copy(tmp / "tokenizer.json", out / "labels_tokenizer.json")
    else:
        class UniGraph(torch.nn.Module):
            def __init__(s, c): super().__init__(); s.c = c
            def forward(s, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask):
                o = s.c(input_ids=input_ids, attention_mask=attention_mask, words_mask=words_mask, text_lengths=text_lengths,
                        span_idx=span_idx, span_mask=span_mask)
                return o.logits if hasattr(o, "logits") else o[0]
        torch.onnx.export(UniGraph(core), tuple(batch[k] for k in base), str(out / "model.onnx"),
                          input_names=base, output_names=["logits"], dynamic_axes=axes, opset_version=19, dynamo=False)
    m.config.to_json_file(str(out / "gliner_config.json"))
    tmp = out / "_tok"; m.data_processor.transformer_tokenizer.save_pretrained(tmp)
    shutil.copy(tmp / "tokenizer.json", out / "tokenizer.json")
    print(f"exported {model_id} → {out} ({'bi' if bi else 'uni'}-encoder)")
    return m

MODELS = os.path.join(WORK, "models")
py_linker = export_gliner(LINKER_MODEL, f"{MODELS}/linker")
if NER_MODEL: export_gliner(NER_MODEL, f"{MODELS}/ner")
if RERANK_MODEL: export_gliner(RERANK_MODEL, f"{MODELS}/rerank")

# ── 3. Lean4 formal guarantees ────────────────────────────────────────────────────────────
sh(f"cd {CRATE}/lean && lean GLinker.lean")
print("Lean4: all theorems in lean/GLinker.lean type-check ✔")

# ── 4. cargo test + release build ─────────────────────────────────────────────────────────
sh(f"cd {CRATE} && cargo test --release 2>&1 | grep -E 'test result|FAILED|panicked'", env=ENV)
sh(f"cd {CRATE} && cargo build --release", env=ENV, quiet=True)
BIN = f"{CRATE}/target/release/glinker"

# ── 5. entity DB, gold set, configs ───────────────────────────────────────────────────────
ENTS = [
    {"entity_id": "Q7240673", "label": "insulin", "aliases": ["INS"], "description": "Peptide hormone that regulates blood glucose", "entity_type": "protein", "popularity": 90},
    {"entity_id": "Q14818098", "label": "TP53", "aliases": ["p53"], "description": "Tumor suppressor gene on chromosome 17", "entity_type": "gene", "popularity": 85},
    {"entity_id": "Q17487737", "label": "BRCA1", "aliases": [], "description": "DNA repair gene linked to hereditary breast cancer", "entity_type": "gene", "popularity": 80},
    {"entity_id": "Q12206", "label": "diabetes mellitus", "aliases": ["diabetes"], "description": "Metabolic disease with high blood sugar", "entity_type": "disease", "popularity": 88},
    {"entity_id": "Q128581", "label": "breast cancer", "aliases": [], "description": "Cancer that develops from breast tissue", "entity_type": "disease", "popularity": 86},
    {"entity_id": "Q19484", "label": "metformin", "aliases": [], "description": "First-line oral medication for type 2 diabetes", "entity_type": "drug", "popularity": 70},
    {"entity_id": "Q37525", "label": "glucose", "aliases": ["blood sugar"], "description": "Simple sugar, main energy source of cells", "entity_type": "chemical", "popularity": 75},
    {"entity_id": "Q35694", "label": "jaguar", "aliases": [], "description": "Large wild cat native to the Americas", "entity_type": "animal", "popularity": 60},
    {"entity_id": "Q30055", "label": "Jaguar Cars", "aliases": ["Jaguar"], "description": "British luxury car manufacturer", "entity_type": "company", "popularity": 65},
]
GOLD_SRC = [
    ("Insulin lowers glucose in patients with diabetes.", [("Insulin", "Q7240673"), ("glucose", "Q37525"), ("diabetes", "Q12206")]),
    ("TP53 and BRCA1 mutations raise the risk of breast cancer.", [("TP53", "Q14818098"), ("BRCA1", "Q17487737"), ("breast cancer", "Q128581")]),
    ("Metformin is prescribed for diabetes mellitus.", [("Metformin", "Q19484"), ("diabetes mellitus", "Q12206")]),
    ("The jaguar hunted near the river at night.", [("jaguar", "Q35694")]),
    ("Jaguar unveiled a new electric sedan in Coventry.", [("Jaguar", "Q30055")]),
    ("Low blood sugar can follow an insulin overdose.", [("blood sugar", "Q37525"), ("insulin", "Q7240673")]),
]
def gold_doc(text, items):
    out, pos = [], 0
    for surf, eid in items:
        s = text.index(surf, pos); out.append({"start": s, "end": s + len(surf), "entity_id": eid}); pos = s + len(surf)
    return {"text": text, "entities": out}
GOLD = [gold_doc(t, it) for t, it in GOLD_SRC]
texts = [g["text"] for g in GOLD]
os.makedirs(f"{WORK}/data", exist_ok=True)
open(f"{WORK}/data/entities.jsonl", "w").write("\n".join(json.dumps(e) for e in ENTS))
open(f"{WORK}/data/gold.jsonl", "w").write("\n".join(json.dumps(g) for g in GOLD))
TEMPLATE = "{label}: {description}"
l3 = {"model_dir": "models/linker", "model_id": LINKER_MODEL, "threshold": THRESHOLD}
l1 = ({"type": "gliner", "model_dir": "models/ner", "labels": ["gene", "protein", "disease", "drug", "chemical", "animal", "company"], "threshold": 0.3}
      if NER_MODEL else {"type": "gazetteer"})
CONFIGS = {
    "simple":   {"name": "glinker-simple", "template": TEMPLATE, "entities": "data/entities.jsonl", "precompute_embeddings": True, "l1": {"type": "none"}, "l3": l3},
    "spans":    {"name": "glinker-spans", "template": TEMPLATE, "entities": "data/entities.jsonl", "precompute_embeddings": True, "l1": {"type": "gazetteer"}, "l2": {"mode": "all"}, "l3": l3},
    "pipeline": {"name": "glinker-rs", "template": TEMPLATE, "entities": "data/entities.jsonl", "precompute_embeddings": True, "l1": l1,
                 "l2": {"search_mode": ["exact", "fuzzy"], "fuzzy": {"min_similarity": 0.85}}, "l3": l3,
                 **({"l4": {"model_dir": "models/rerank", "threshold": THRESHOLD, "max_labels": 20}} if RERANK_MODEL else {})},
}
for k, c in CONFIGS.items(): json.dump(c, open(f"{WORK}/{k}.json", "w"), indent=1)

def rs_link(cfg, texts):
    return json.loads(sh(f"{BIN} link {WORK}/{cfg}.json " + " ".join(json.dumps(t) for t in texts), env=ENV, quiet=True))

# ── 6. parity: Rust (orp/ort) vs Python GLiNER on the same weights ─────────────────────────
labels = [TEMPLATE.format(**e) for e in ENTS]
emb = py_linker.encode_labels(labels)
lab2id = {l: e["entity_id"] for l, e in zip(labels, ENTS)}
def py_preds(text, spans=None):
    kw = {"input_spans": [[{"start": s, "end": e} for s, e in spans]]} if spans else {}
    r = py_linker.predict_with_embeds(text, emb, labels, threshold=THRESHOLD, flat_ner=True, multi_label=False, **kw)
    return sorted((p["start"], p["end"], lab2id[p["label"]], round(float(p["score"]), 3)) for p in r)
def rs_preds(doc):
    return sorted((x["mention_start"], x["mention_end"], x["linked_entity"]["entity_id"], round(x["linked_entity"]["confidence"], 3))
                  for x in doc if x["linked_entity"])
def compare(name, a, b, tol=2e-3):
    same = len(a) == len(b) and all(p[:3] == q[:3] and abs(p[3] - q[3]) <= tol for p, q in zip(a, b))
    print(f"  [{'MATCH' if same else 'DIFF '}] {name}\n     py: {a}\n     rs: {b}"); return same
ok = True
print("Parity A — all spans (simple pipeline):")
out = rs_link("simple", texts)
for t, d in zip(texts, out["entities"]): ok &= compare(t, py_preds(t), rs_preds(d))
print("Parity B — L1 input_spans constraint:")
out = rs_link("spans", texts)
for t, d in zip(texts, out["entities"]):
    spans = [(x["mention_start"], x["mention_end"]) for x in d]
    ok &= compare(t, py_preds(t, spans), rs_preds(d))
print("PARITY:", "PASS ✔" if ok else "DIFF — see above (scores within ±2e-3 count as equal)")

# ── 7. MCC evaluation (Rust) + sklearn cross-check ────────────────────────────────────────
from sklearn.metrics import matthews_corrcoef
rep = json.loads(sh(f"{BIN} eval {WORK}/pipeline.json {WORK}/data/gold.jsonl", env=ENV, quiet=True))
print(json.dumps(rep, indent=1))
out = rs_link("pipeline", texts)
yt, yp = [], []
for g, d in zip(GOLD, out["entities"]):
    gm = {(e["start"], e["end"]): e["entity_id"] for e in g["entities"]}
    for x in d:
        yt.append(gm.pop((x["mention_start"], x["mention_end"]), "NIL")); yp.append(x["linked_entity"]["entity_id"] if x["linked_entity"] else "NIL")
    for v in gm.values(): yt.append(v); yp.append("NIL")
sk = matthews_corrcoef(yt, yp)
print(f"MCC mention-level  rust={rep['mcc_mention']:.6f}  sklearn={sk:.6f}  {'✔ agree' if abs(sk - rep['mcc_mention']) < 1e-9 else '✘ mismatch'}")
print(f"MCC candidate-level rust={rep['mcc_candidate']:.6f}")

# ── 8. OpenAI-compatible /v1/chat/completions ─────────────────────────────────────────────
srv = subprocess.Popen([BIN, "serve", f"{WORK}/pipeline.json", f"0.0.0.0:{PORT}"], env={**os.environ, **ENV},
                       stdout=subprocess.DEVNULL, stderr=open(f"{WORK}/serve.log", "w"))
import urllib.request
for _ in range(120):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
    except Exception: time.sleep(0.5)
from openai import OpenAI
client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="not-needed")
r = client.chat.completions.create(model="glinker-rs", messages=[
    {"role": "system", "content": "Link biomedical entities."},
    {"role": "user", "content": "Jaguar sponsors a diabetes charity; the jaguar is endangered."}])
res = json.loads(r.choices[0].message.content)
for e in res["entities"][0]:
    le = e["linked_entity"] or {}
    print(f"  {e['mention_text']!r:16} [{e['mention_start']}:{e['mention_end']}] → {le.get('entity_id', 'NIL'):10} {le.get('label', ''):18} p={le.get('confidence', 0):.3f}")
print("stats:", res["stats"], "| usage:", r.usage)
print(f"\nServer running (pid {srv.pid}) at http://127.0.0.1:{PORT}/v1/chat/completions — srv.terminate() to stop.")
print("Training: none — this port is inference-only. Any fine-tuning of the linker must use SAM (Sharpness-Aware Minimization) before re-export.")
