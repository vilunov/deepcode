use lazy_static::lazy_static;
use tree_sitter::*;

extern "C" {
    fn tree_sitter_go() -> Language;
    fn tree_sitter_java() -> Language;
    fn tree_sitter_javascript() -> Language;
    fn tree_sitter_php() -> Language;
    fn tree_sitter_python() -> Language;
    fn tree_sitter_ruby() -> Language;
}

lazy_static! {
    pub static ref LANG_GO: Language = unsafe { tree_sitter_go() };
    pub static ref LANG_JAVA: Language = unsafe { tree_sitter_java() };
    pub static ref LANG_JAVASCRIPT: Language = unsafe { tree_sitter_javascript() };
    pub static ref LANG_PHP: Language = unsafe { tree_sitter_php() };
    pub static ref LANG_PYTHON: Language = unsafe { tree_sitter_python() };
    pub static ref LANG_RUBY: Language = unsafe { tree_sitter_ruby() };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go() {
        let mut parser = Parser::new();
        parser.set_language(*LANG_GO).unwrap();
        let sample = r#"
func main() {
    fmt.Println("hello world")
}"#;
        let result: Tree = parser.parse(sample, None).unwrap();
        let mut cursor = result.walk();
        let node = result.root_node();
        assert_eq!(node.kind(), "source_file");
        let children: Vec<_> = node.children(&mut cursor).collect();
        assert_eq!(children.len(), 1);
        let func_decl = &children[0];
        assert_eq!(func_decl.kind(), "function_declaration");
        let children: Vec<_> = func_decl.children(&mut cursor).collect();
        assert_eq!(children[0].kind(), "func");
        assert_eq!(children[1].kind(), "identifier");
        assert_eq!(children[2].kind(), "parameter_list");
        assert_eq!(children[3].kind(), "block");
    }
}
