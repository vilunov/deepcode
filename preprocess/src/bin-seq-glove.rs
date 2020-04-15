use glob::glob;
use libflate::gzip::Decoder;
use serde_json::Deserializer;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::LineWriter;

mod structs;
mod utils;
use crate::structs::*;
use crate::utils::*;

fn get_words() -> Vec<String> {
    let filename = "../resources/glove.6B.100d.txt";
    let contents = fs::read_to_string(filename).unwrap();
    let lines = contents.split("\n");
    let mut words = lines
        .map(|line| line.split(" ").into_iter().next().unwrap().to_owned())
        .collect::<Vec<String>>();
    let _ = words.pop();
    dbg!(words.len());
    words
}

fn write_to_file(w: &[String]) {
    let file = File::create("../cache/glove-vocab.txt").unwrap();
    let mut file = LineWriter::new(file);
    for i in w {
        file.write_all(i.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
    }
}

fn build_inverse(w: &[String]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, j) in w.iter().enumerate() {
        map.insert(j.to_owned(), i);
    }
    map
}

fn convert_to_h5(
    train: &mut hdf5::File,
    valid: &mut hdf5::File,
    test: &mut hdf5::File,
    lang: &'static str,
    doc_vocab: &HashMap<String, usize>,
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
            .flat_map(|i| doc_vocab.get(i.as_str()).map(|&i| i as u32).into_iter())
            .collect::<Vec<u32>>();
        let name_vec = split_identifier(&snippet.func_name)
            .into_iter()
            .flat_map(|i| doc_vocab.get(i.as_str()).map(|&i| i as u32).into_iter())
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

pub fn build_h5(doc_vocab: &HashMap<String, usize>) {
    let mut train = hdf5::File::open("../cache/data/glove/train.h5", "w").unwrap();
    let mut valid = hdf5::File::open("../cache/data/glove/valid.h5", "w").unwrap();
    let mut test = hdf5::File::open("../cache/data/glove/test.h5", "w").unwrap();
    for lang in LANGS {
        convert_to_h5(&mut train, &mut valid, &mut test, lang, doc_vocab);
    }
}

fn main() {
    let words = get_words();
    let vocab = build_inverse(words.as_slice());
    write_to_file(words.as_slice());
    build_h5(&vocab);
}
