# glinker-rs

Rust port of [Knowledgator/GLinker](https://github.com/Knowledgator/GLinker) (zero-shot entity linking)
built on [fbilhaut/orp](https://github.com/fbilhaut/orp) 1.0 + [ort](https://ort.pyke.io) 2.0.0-rc.9.

```
text ─L1─▶ mentions ─L2─▶ candidates ─L3 (bi-encoder) / L4 (uni-encoder rerank)─▶ links ─L0─▶ L0Output
       gazetteer |        dict layer               orp::Pipeline: SpanPipeline          aggregation
       GLiNER NER        exact + fuzzy             + LabelEncoderPipeline               + stats
```

| Python (glinker) | Rust |
|---|---|
| `l1` spaCy / GLiNER | `l1::L1::{Gliner, Gazetteer, Provided}` |
| `l2` DictLayer, exact + rapidfuzz | `l2::{DictLayer, L2}`, `fuzz::ratio` (Indel ratio, bit-identical to rapidfuzz) |
| `l3` GLiNER `predict_with_embeds(input_spans=…)` | `l3::L3` → `gliner::pipeline::SpanPipeline` |
| `encode_labels` / precomputed embeddings | `gliner::labels::LabelEncoderPipeline`, `Executor::precompute_embeddings` |
| `l4` reranker with label chunking | `l4::L4` |
| `l0` aggregation + stats | `l0` |
| `ProcessorFactory`/`DAGExecutor` | `config::PipelineConfig` (JSON) + `Executor` |

Extra: `metrics` (MCC, candidate- and mention-level), `openai` (`/v1/chat/completions`), `lean/GLinker.lean`.

## Build / run
```bash
export ORT_LIB_LOCATION=$(python -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__)+'/capi')")
export ORT_DYLIB_PATH=$(ls $ORT_LIB_LOCATION/libonnxruntime.so*|tail -1)     # onnxruntime==1.20.x
cargo build --release
./target/release/glinker link  pipeline.json "TP53 mutations cause cancer."
./target/release/glinker eval  pipeline.json gold.jsonl
./target/release/glinker serve pipeline.json 0.0.0.0:8000
```
Model dirs are produced by the notebook exporter: `model.onnx`, `tokenizer.json`, `gliner_config.json`
(+ `labels_encoder.onnx`, `labels_tokenizer.json` for bi-encoders).

## Not ported
spaCy L1 (no Rust equivalent → gazetteer or GLiNER L1), Redis/Elasticsearch/Postgres layers
(implement `l2::CandidateLayer`), L3 `schema.ranking`, non-whitespace GLiNER word splitters,
token-mode (non-span) GLiNER models.
