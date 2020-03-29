use tree_sitter::{Language, Parser};

mod parsers;
use crate::parsers::*;

fn main() {
    let mut parser = Parser::new();
    parser
        .set_language(*LANG_JAVASCRIPT)
        .expect("Failed to set parser's language");
    let src = "function a() {}";
    let res = parser.parse(src, None).unwrap();
    dbg!(res);
}
