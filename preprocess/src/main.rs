use glob::glob;
use libflate::gzip::Decoder;
use serde_json::Deserializer;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::tokenizer::Model;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

mod structs;
use crate::structs::*;

const LANGS: &[&'static str] = &["go", "java", "javascript", "php", "python", "ruby"];

fn train_glob_pattern(language: &'static str) -> String {
    format!(
        "../resources/data/{}/final/jsonl/train/**/*.jsonl.gz",
        language
    )
}

fn valid_test_glob_pattern(language: &'static str) -> (String, String) {
    let valid = format!(
        "../resources/data/{}/final/jsonl/valid/**/*.jsonl.gz",
        language
    );
    let test = format!(
        "../resources/data/{}/final/jsonl/test/**/*.jsonl.gz",
        language
    );
    (valid, test)
}

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
    let trainer = BpeTrainer::new(0, 32767);
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
        }
    }
    let trainer = BpeTrainer::new(0, 32767);
    let model = trainer.train(counter).unwrap();
    model.save(Path::new("../cache/vocabs"), name).unwrap();
}

pub fn build_vocabs() {
    for lang in LANGS {
        process_code(lang);
    }
    process_docs("doc");
}

fn convert_to_h5(
    train: &mut hdf5::File,
    valid: &mut hdf5::File,
    test: &mut hdf5::File,
    lang: &'static str,
) {
    dbg!("convert_to_h5", lang);
    let train_pattern = train_glob_pattern(lang);
    let (valid_pattern, test_pattern) = valid_test_glob_pattern(lang);

    let vocab_code = BPE::from_files(
        &format!("../cache/vocabs/code-{}-vocab.json", lang),
        &format!("../cache/vocabs/code-{}-merges.txt", lang),
    )
    .unwrap()
    .build()
    .unwrap();

    let vocab_doc = BPE::from_files(
        "../cache/vocabs/doc-vocab.json",
        "../cache/vocabs/doc-merges.txt",
    )
    .unwrap()
    .build()
    .unwrap();

    let convert_snippet = |snippet: SnippetBoth| {
        let code_vec = snippet
            .code_tokens
            .into_iter()
            .flat_map(|i| vocab_code.token_to_id(&i))
            .collect::<Vec<_>>();
        let doc_vec = snippet
            .docstring_tokens
            .into_iter()
            .flat_map(|i| vocab_doc.token_to_id(&i))
            .collect::<Vec<_>>();
        let mut code_tokens = [0; MAX_LEN];
        let mut doc_tokens = [0; MAX_LEN];
        let code_len = code_vec.len().min(MAX_LEN);
        let doc_len = doc_vec.len().min(MAX_LEN);
        code_tokens[..code_len].copy_from_slice(&code_vec[..code_len]);
        doc_tokens[..doc_len].copy_from_slice(&doc_vec[..doc_len]);
        Snippet {
            code_len,
            code_tokens,
            doc_len,
            doc_tokens,
        }
    };

    let process = |glob_pattern: &str, file: &mut hdf5::File| {
        let snippets = glob(glob_pattern)
            .expect("Failed to read glob pattern")
            .flat_map(|entry| {
                let path = entry.expect("Failed to get entry");
                let file = File::open(path).expect("Failed to open file");
                let decoder = Decoder::new(file).expect("Failed to create decoder");
                Deserializer::from_reader(decoder)
                    .into_iter::<SnippetBoth>()
                    .map(Result::unwrap)
                    .map(convert_snippet)
                    .filter(|snippet| snippet.code_len > 0 && snippet.doc_len > 0)
            })
            .collect::<Vec<_>>();
        dbg!(snippets.len());
        let dataset = file
            .new_dataset::<Snippet>()
            .create(lang, snippets.len())
            .unwrap();
        dataset.write(&snippets[..]).unwrap();
    };

    process(&train_pattern, train);
    process(&valid_pattern, valid);
    process(&test_pattern, test);
}

pub fn build_h5() {
    let mut train = hdf5::File::open("../cache/data/train.h5", "w").unwrap();
    let mut valid = hdf5::File::open("../cache/data/valid.h5", "w").unwrap();
    let mut test = hdf5::File::open("../cache/data/test.h5", "w").unwrap();
    for lang in LANGS {
        convert_to_h5(&mut train, &mut valid, &mut test, lang);
    }
}

fn main() {
    build_vocabs();
    build_h5();
}
