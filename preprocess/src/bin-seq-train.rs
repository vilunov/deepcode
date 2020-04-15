use glob::glob;
use libflate::gzip::Decoder;
use serde_json::Deserializer;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};

use std::fs::File;

mod structs;
mod utils;
use crate::structs::*;
use crate::utils::*;

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
    let tokenizer_code = Tokenizer::new(Box::new(vocab_code));

    let vocab_doc = BPE::from_files(
        "../cache/vocabs/doc-vocab.json",
        "../cache/vocabs/doc-merges.txt",
    )
    .unwrap()
    .build()
    .unwrap();
    let tokenizer_doc = Tokenizer::new(Box::new(vocab_doc));

    let convert_snippet = |snippet: SnippetBoth| {
        let code_vec: Vec<u32> = snippet
            .code_tokens
            .into_iter()
            .flat_map(|i| {
                tokenizer_code
                    .encode(EncodeInput::Single(i))
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<u32>>();
        let doc_vec: Vec<u32> = snippet
            .docstring_tokens
            .into_iter()
            .flat_map(|i| {
                tokenizer_doc
                    .encode(EncodeInput::Single(i))
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<u32>>();
        let name_vec = split_identifier(&snippet.func_name)
            .into_iter()
            .flat_map(|i| {
                tokenizer_doc
                    .encode(EncodeInput::Single(i))
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<u32>>();
        let mut code_tokens = [0; MAX_LEN];
        let mut doc_tokens = [0; MAX_LEN];
        let mut name_tokens = [0; MAX_NAME_LEN];
        let code_len = code_vec.len().min(MAX_LEN);
        let doc_len = doc_vec.len().min(MAX_LEN);
        let name_len = name_vec.len().min(MAX_NAME_LEN);
        code_tokens[..code_len].copy_from_slice(&code_vec[..code_len]);
        doc_tokens[..doc_len].copy_from_slice(&doc_vec[..doc_len]);
        name_tokens[..name_len].copy_from_slice(&name_vec[..name_len]);
        Snippet {
            code_len,
            code_tokens,
            doc_len,
            doc_tokens,
            name_len,
            name_tokens,
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
                    .filter(|snippet| {
                        snippet.code_len > 0 && snippet.doc_len > 0 && snippet.name_len > 0
                    })
            })
            .collect::<Vec<_>>();
        dbg!(snippets.len());
        let dataset = file
            .new_dataset::<Snippet>()
            .gzip(6)
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
    build_h5();
}
