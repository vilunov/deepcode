use glob::glob;
use libflate::gzip::Decoder;
use serde_json::Deserializer;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::tokenizer::{EncodeInput, Model, Tokenizer};

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

mod structs;
mod utils;
use crate::structs::*;
use crate::utils::*;

pub const LANGS: &[&'static str] = &["go", "java", "javascript", "php", "python", "ruby"];

fn main() {
    let mut search = hdf5::File::open("../cache/data/search.h5", "w").unwrap();
    for lang in LANGS {
        convert_to_h5(&mut search, lang);
    }
}

fn convert_to_h5(file: &mut hdf5::File, lang: &'static str) {
    dbg!("convert_to_h5", lang);
    let pattern = format!("../cache/data-test/{}.jsonl.gz", lang);

    let vocab_code = BPE::from_files(
        &format!("../cache/vocabs/code-{}-vocab.json", lang),
        &format!("../cache/vocabs/code-{}-merges.txt", lang),
    )
        .unwrap()
        .build()
        .unwrap();
    let tokenizer_code = Tokenizer::new(Box::new(vocab_code));

    let convert_snippet = |snippet: SnippetSearch| {
        let code_vec: Vec<u32> = snippet
            .function_tokens
            .into_iter()
            .flat_map(|i| {
                tokenizer_code
                    .encode(EncodeInput::Single(i))
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<u32>>();
        let mut code_tokens = [0; MAX_LEN2];
        let code_len = code_vec.len().min(MAX_LEN2);
        code_tokens[..code_len].copy_from_slice(&code_vec[..code_len]);
        Snippet2 {
            code_len,
            code_tokens,
        }
    };

    let snippets = glob(pattern.as_ref())
        .expect("Failed to read glob pattern")
        .flat_map(|entry| {
            let path = entry.expect("Failed to get entry");
            let file = File::open(path).expect("Failed to open file");
            let decoder = Decoder::new(file).expect("Failed to create decoder");
            Deserializer::from_reader(decoder)
                .into_iter::<SnippetSearch>()
                .map(Result::unwrap)
                .map(convert_snippet)
        })
        .collect::<Vec<_>>();
    dbg!(snippets.len());
    let dataset = file
        .new_dataset::<Snippet2>()
        .create(lang, snippets.len())
        .unwrap();
    dataset.write(&snippets[..]).unwrap();
}
