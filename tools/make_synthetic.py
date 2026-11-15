"""Build a synthetic GLiNER bi-encoder (same tensor contract as GLiNER's
`export_to_onnx(from_labels_embeddings=True)`) + label encoder + tokenizers, for offline E2E tests.

text graph : input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask, labels_embeddings -> logits [B,L,K,C]
label graph: input_ids, attention_mask -> embeddings [N,H]  (masked mean pool)
"""
import json, os, sys
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from tokenizers import Tokenizer, models, pre_tokenizers, processors

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", ":", ".", ",", "insulin", "glucose", "tp53", "cancer", "diabetes",
         "regulates", "mutations", "cause", "and", "the", "hormone", "gene", "disease", "tumor", "suppressor",
         "sugar", "metabolic", "peptide", "blood", "in", "is", "a", "patients", "with", "levels", "treat"]
vocab = {w: i for i, w in enumerate(words)}
tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
from tokenizers import normalizers
tok.normalizer = normalizers.Lowercase()
tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
tok.save(os.path.join(out, "tokenizer.json"))
tok.save(os.path.join(out, "labels_tokenizer.json"))

H = 64
rng = np.random.default_rng(0)
E = rng.normal(size=(len(words), H)).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
# make "type" words (gene/disease/hormone) align with their instances, so NER labels work too
E[vocab["gene"]] = E[vocab["tp53"]]
E[vocab["disease"]] = (E[vocab["cancer"]] + E[vocab["diabetes"]]) / np.sqrt(2)
E[vocab["hormone"]] = E[vocab["insulin"]]
for w in ["[PAD]", "[CLS]", "[SEP]", ":", ".", ","]:
    E[vocab[w]] *= 0.0
# description words: small, so label "insulin: Peptide hormone ..." ≈ insulin direction
for w in ["tumor", "suppressor", "sugar", "metabolic", "peptide", "blood"]:
    E[vocab[w]] *= 0.1

def C(name, arr):
    return nh.from_array(np.asarray(arr), name)

# ---------------- text graph ----------------
n = []
inits = [C("E", E), C("one", np.array([1], np.int64)), C("zero", np.array(0, np.int64)), C("one_s", np.array(1, np.int64)),
         C("ax0", np.array([0], np.int64)), C("ax1", np.array([1], np.int64)), C("ax2", np.array([2], np.int64)),
         C("neg1", np.array([-1], np.int64)), C("scale", np.array(30.0, np.float32)), C("bias", np.array(5.0, np.float32)),
         C("big", np.array(-1e4, np.float32)), C("half", np.array(0.5, np.float32)),
         C("idx0", np.array([0], np.int64)), C("idx1", np.array([1], np.int64))]
n += [h.make_node("Gather", ["E", "input_ids"], ["tok"], axis=0)]                       # B,T,H
n += [h.make_node("ReduceMax", ["text_lengths"], ["Lk"], keepdims=0)]                   # scalar
n += [h.make_node("Add", ["Lk", "one_s"], ["Lp1"])]
n += [h.make_node("Range", ["one_s", "Lp1", "one_s"], ["rng"])]                         # [L] 1..L
n += [h.make_node("Unsqueeze", ["rng", "ax0"], ["rng1"]), h.make_node("Unsqueeze", ["rng1", "neg1"], ["rng2"])]  # 1,L,1
n += [h.make_node("Unsqueeze", ["words_mask", "ax1"], ["wm1"])]                         # B,1,T
n += [h.make_node("Equal", ["wm1", "rng2"], ["Wb"]), h.make_node("Cast", ["Wb"], ["W"], to=T.FLOAT)]  # B,L,T
n += [h.make_node("MatMul", ["W", "tok"], ["wrep"])]                                    # B,L,H
# span one-hots
n += [h.make_node("Range", ["zero", "Lk", "one_s"], ["rng0"])]                          # 0..L-1
n += [h.make_node("Unsqueeze", ["rng0", "ax0"], ["r0a"]), h.make_node("Unsqueeze", ["r0a", "ax0"], ["r0b"])]  # 1,1,L
n += [h.make_node("Gather", ["span_idx", "zero"], ["st"], axis=2), h.make_node("Gather", ["span_idx", "one_s"], ["en"], axis=2)]  # B,N
n += [h.make_node("Unsqueeze", ["st", "neg1"], ["st1"]), h.make_node("Unsqueeze", ["en", "neg1"], ["en1"])]   # B,N,1
n += [h.make_node("Equal", ["st1", "r0b"], ["So"]), h.make_node("Cast", ["So"], ["Sf"], to=T.FLOAT)]
n += [h.make_node("Equal", ["en1", "r0b"], ["Eo"]), h.make_node("Cast", ["Eo"], ["Ef"], to=T.FLOAT)]
n += [h.make_node("MatMul", ["Sf", "wrep"], ["srep"]), h.make_node("MatMul", ["Ef", "wrep"], ["erep"])]      # B,N,H
n += [h.make_node("Add", ["srep", "erep"], ["sum"]), h.make_node("Mul", ["sum", "half"], ["rep"])]
# a span is only good if start==end-ish: penalise width via (start-end)^2
n += [h.make_node("Sub", ["en", "st"], ["wd"]), h.make_node("Cast", ["wd"], ["wdf"], to=T.FLOAT),
      h.make_node("Mul", ["wdf", "wdf"], ["wd2"]), h.make_node("Unsqueeze", ["wd2", "neg1"], ["wd3"])]
n += [h.make_node("Transpose", ["labels_embeddings"], ["LT"], perm=[1, 0])]              # H,C
n += [h.make_node("MatMul", ["rep", "LT"], ["dots"])]                                   # B,N,C
n += [h.make_node("Mul", ["dots", "scale"], ["sc"]), h.make_node("Sub", ["sc", "bias"], ["sc2"]),
      h.make_node("Mul", ["wd3", "scale"], ["pen"]), h.make_node("Sub", ["sc2", "pen"], ["sc3"])]
n += [h.make_node("Unsqueeze", ["span_mask", "neg1"], ["sm1"]),
      h.make_node("Where", ["sm1", "sc3", "big"], ["masked"])]
n += [h.make_node("Shape", ["input_ids"], ["shp"]), h.make_node("Slice", ["shp", "idx0", "idx1"], ["Bd"]),
      h.make_node("Unsqueeze", ["Lk", "ax0"], ["Ld"]), h.make_node("Shape", ["labels_embeddings"], ["lshp"]),
      h.make_node("Slice", ["lshp", "idx0", "idx1"], ["Cd"]),
      h.make_node("Concat", ["Bd", "Ld", "neg1", "Cd"], ["oshape"], axis=0),
      h.make_node("Reshape", ["masked", "oshape"], ["logits"])]
g = h.make_graph(n, "synthetic_gliner_bi",
    [h.make_tensor_value_info("input_ids", T.INT64, ["B", "T"]),
     h.make_tensor_value_info("attention_mask", T.INT64, ["B", "T"]),
     h.make_tensor_value_info("words_mask", T.INT64, ["B", "T"]),
     h.make_tensor_value_info("text_lengths", T.INT64, ["B", 1]),
     h.make_tensor_value_info("span_idx", T.INT64, ["B", "N", 2]),
     h.make_tensor_value_info("span_mask", T.BOOL, ["B", "N"]),
     h.make_tensor_value_info("labels_embeddings", T.FLOAT, ["C", "H"])],
    [h.make_tensor_value_info("logits", T.FLOAT, ["B", "L", "K", "C"])], inits)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, os.path.join(out, "model.onnx"))

# ---------------- label encoder ----------------
n = [h.make_node("Gather", ["E", "input_ids"], ["t"], axis=0),
     h.make_node("Cast", ["attention_mask"], ["mf"], to=T.FLOAT),
     h.make_node("Unsqueeze", ["mf", "neg1"], ["m3"]),
     h.make_node("Mul", ["t", "m3"], ["tm"]),
     h.make_node("ReduceSum", ["tm", "ax1"], ["s"], keepdims=0),
     h.make_node("ReduceSum", ["mf", "ax1"], ["cnt"], keepdims=1),
     h.make_node("Max", ["cnt", "eps"], ["cnt2"]),
     h.make_node("Div", ["s", "cnt2"], ["mean"]),
     h.make_node("Mul", ["mean", "three"], ["embeddings"])]  # ×3 undoes [CLS]/[SEP] dilution for 1-word labels
g = h.make_graph(n, "synthetic_label_encoder",
    [h.make_tensor_value_info("input_ids", T.INT64, ["N", "T"]), h.make_tensor_value_info("attention_mask", T.INT64, ["N", "T"])],
    [h.make_tensor_value_info("embeddings", T.FLOAT, ["N", H])],
    [C("E", E), C("ax1", np.array([1], np.int64)), C("neg1", np.array([-1], np.int64)), C("eps", np.array(1e-9, np.float32)), C("three", np.array(3.0, np.float32))])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, os.path.join(out, "labels_encoder.onnx"))

json.dump({"max_width": 12, "max_len": 384, "labels_encoder": "synthetic-labels", "span_mode": "markerV0"},
          open(os.path.join(out, "gliner_config.json"), "w"))
print("ok", out)
