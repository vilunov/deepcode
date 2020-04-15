use std::path::PathBuf;

fn compile(language: &str) {
    let dirname = format!("tree-sitter-{}", language);
    let dir: PathBuf = ["parsers", &dirname, "src"].iter().collect();
    let files = [dir.join("parser.c"), dir.join("scanner.c")];
    cc::Build::new()
        .include(&dir)
        .warnings(false)
        .files(files.iter().filter(|f| f.exists()))
        .compile(&dirname);
}

fn main() {
    let languages = ["go", "java", "javascript", "php", "python", "ruby"];
    for language in languages.iter() {
        compile(language);
    }
}
