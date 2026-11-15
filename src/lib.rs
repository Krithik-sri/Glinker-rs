//! # glinker-rs
//!
//! Rust port of [Knowledgator GLinker](https://github.com/Knowledgator/GLinker) built on
//! [`orp`](https://github.com/fbilhaut/orp) (ONNX Runtime pipelines) and [`ort`](https://ort.pyke.io).
//!
//! Layer map (same names as the Python package):
//!
//! | Layer | Python                          | Rust module   |
//! |-------|---------------------------------|---------------|
//! | L1    | spaCy / GLiNER mention detection| [`l1`]        |
//! | L2    | Dict/Redis/ES/PG candidate search | [`l2`] (dict layer, exact + fuzzy) |
//! | L3    | GLiNER bi-encoder linking       | [`l3`] + [`gliner`] |
//! | L4    | GLiNER uni-encoder reranking    | [`l4`] + [`gliner`] |
//! | L0    | aggregation                     | [`l0`]        |
//!
//! Every model call goes through an `orp::pipeline::Pipeline` implementation
//! ([`gliner::pipeline::SpanPipeline`], [`gliner::labels::LabelEncoderPipeline`]).

pub mod types;
pub mod text;
pub mod template;
pub mod fuzz;
pub mod gliner;
pub mod l1;
pub mod l2;
pub mod l3;
pub mod l4;
pub mod l0;
pub mod config;
pub mod executor;
pub mod metrics;
pub mod openai;

pub type Result<T> = orp::Result<T>;

pub use executor::Executor;
pub use types::*;
