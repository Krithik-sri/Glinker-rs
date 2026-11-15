//! CLI:
//!   glinker link  <config.json> "text" ["text" ...]   → L0 JSON on stdout
//!   glinker eval  <config.json> <gold.jsonl>          → MCC report JSON
//!   glinker serve <config.json> [addr=0.0.0.0:8000]   → OpenAI-compatible server
//!
//! ONNX Runtime is loaded from `ORT_DYLIB_PATH` (feature `load-dynamic`, the default).

use std::sync::{Arc, Mutex};
use glinker_rs::metrics::{evaluate, GoldDoc};
use glinker_rs::openai::{serve, AppState};
use glinker_rs::Executor;

fn usage() -> ! {
    eprintln!("usage: glinker <link|eval|serve> <config.json> [...]");
    std::process::exit(2)
}

fn main() -> glinker_rs::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }
    let mut ex = Executor::from_config_file(&args[2])?;
    match args[1].as_str() {
        "link" => {
            let texts: Vec<String> = args[3..].to_vec();
            let out = ex.execute(&texts)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "eval" => {
            let path = args.get(3).unwrap_or_else(|| usage());
            let gold: Vec<GoldDoc> = std::fs::read_to_string(path)?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            let texts: Vec<String> = gold.iter().map(|g| g.text.clone()).collect();
            let out = ex.execute(&texts)?;
            let report = evaluate(&gold, &out.entities);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "serve" => {
            let addr = args.get(3).cloned().unwrap_or_else(|| "0.0.0.0:8000".into());
            let name = if ex.name.is_empty() { "glinker-rs".to_string() } else { ex.name.clone() };
            let state = AppState { executor: Arc::new(Mutex::new(ex)), model_name: name };
            eprintln!("glinker-rs listening on http://{addr}/v1/chat/completions");
            tokio::runtime::Runtime::new()?.block_on(serve(state, &addr))?;
        }
        _ => usage(),
    }
    Ok(())
}
