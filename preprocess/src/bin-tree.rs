use tree_sitter::*;

mod parsers;
use crate::parsers::*;

fn traverse(node: &Node, src: &str) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        dbg!(
            child.utf8_text(src.as_bytes()).unwrap(),
            child.id(),
            child.kind_id()
        );
        traverse(&child, src);
    }
}

fn main() {
    let mut parser = Parser::new();
    dbg!(LANG_JAVASCRIPT.node_kind_count());
    parser
        .set_language(*LANG_JAVASCRIPT)
        .expect("Failed to set parser's language");
    let src = "function a() { console.log(1); /* k */ }";
    let res = parser.parse(src, None).unwrap();
    let node = res.root_node();
    traverse(&node, src)
}
