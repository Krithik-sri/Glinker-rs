
## Repo layout
- `notebooks/glinker_rs_orp.ipynb` — single-cell Colab: export → Lean check → build → parity → MCC → server
- `tools/make_synthetic.py` — synthetic ONNX models with GLiNER's bi-encoder tensor contract (offline E2E)
- `tools/parity.py` — independent numpy/onnxruntime reference for span decoding parity
- `tools/build_nb.py` + `tools/cell_template.py` — regenerate the notebook from this crate
- `tests/fixtures/` — entity DB, gold set, configs for the synthetic E2E

Offline E2E:
```bash
python tools/make_synthetic.py tests/fixtures/models/linker
./target/release/glinker eval tests/fixtures/gazetteer.json tests/fixtures/gold.jsonl
python tools/parity.py tests/fixtures/models/linker tests/fixtures/simple.json target/release/glinker
```
