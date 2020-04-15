use glob::glob;
use libflate::gzip::Decoder;
use serde_json::Deserializer;
use tokenizers::models::bpe::BpeTrainer;
use tokenizers::tokenizer::Model;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

mod structs;
mod utils;
use crate::structs::*;
use crate::utils::*;

fn filter(token: &str) -> bool {
    !token.starts_with("//")
        && !token.starts_with("#")
        && token.len() < 64
        && !token.contains(" ")
        && !token.is_empty()
        && !token.contains("\n")
        && !token.contains("\t")
}

fn process_code(language: &'static str) {
    println!("Running {}", language);
    let glob_pattern = train_glob_pattern(language);
    let name = format!("code-{}", language);
    let mut counter = HashMap::new();
    for entry in glob(&glob_pattern).expect("Failed to read glob pattern") {
        let path = entry.expect("Failed to get entry");
        let file = File::open(path).expect("Failed to open file");
        let decoder = Decoder::new(file).expect("Failed to create decoder");
        let deser = Deserializer::from_reader(decoder);
        for snippet in deser.into_iter::<SnippetCode>().map(Result::unwrap) {
            for token in snippet.code_tokens {
                *counter.entry(token).or_insert(0) += 1;
            }
        }
    }
    counter.retain(|k, _| filter(k.as_str()));
    let trainer = BpeTrainer::new(0, 12000);
    let model = trainer.train(counter).unwrap();
    model.save(Path::new("../cache/vocabs"), &name).unwrap();
}

fn process_docs(name: &'static str) {
    println!("Running {}", name);
    let mut counter = HashMap::new();
    for entry in glob("../resources/data/*/final/jsonl/train/**/*.jsonl.gz")
        .expect("Failed to read glob pattern")
    {
        let path = entry.expect("Failed to get entry");
        let file = File::open(path).expect("Failed to open file");
        let decoder = Decoder::new(file).expect("Failed to create decoder");
        let deser = Deserializer::from_reader(decoder);
        for snippet in deser.into_iter::<SnippetDoc>().map(Result::unwrap) {
            for token in snippet.docstring_tokens {
                *counter.entry(token).or_insert(0) += 1;
            }
            for token in split_identifier(&snippet.func_name) {
                *counter.entry(token).or_insert(0) += 1;
            }
        }
    }
    let trainer = BpeTrainer::new(0, 32000);
    let model = trainer.train(counter).unwrap();
    model.save(Path::new("../cache/vocabs"), name).unwrap();
}

pub fn build_vocabs() {
    for lang in LANGS {
        process_code(lang);
    }
    process_docs("doc");
}

fn main() {
    build_vocabs();
}
