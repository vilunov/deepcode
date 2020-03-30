use lazy_static::lazy_static;
use regex::Regex;
use std::borrow::Borrow;

pub const LANGS: &[&'static str] = &["go", "java", "javascript", "php", "python", "ruby"];

pub fn train_glob_pattern(language: &'static str) -> String {
    format!(
        "../resources/data/{}/final/jsonl/train/**/*.jsonl.gz",
        language
    )
}

pub fn valid_test_glob_pattern(language: &'static str) -> (String, String) {
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

lazy_static! {
    static ref RE1: Regex = Regex::new(r"(?P<last>[A-Z])").unwrap();
    static ref RE2: Regex = Regex::new(r"[^\p{Alphabetic}]").unwrap();
}

pub fn split_identifier(id: &str) -> Vec<String> {
    let pass1 = RE1.replace_all(id, "-$last");
    RE2.split(pass1.borrow())
        .filter(|&i| !i.is_empty())
        .map(|i| i.to_lowercase())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_identifier() {
        assert_eq!(split_identifier("camelCase"), vec!["camel", "case"]);
        assert_eq!(split_identifier("snake_case"), vec!["snake", "case"]);
        assert_eq!(
            split_identifier("strangeMix_of_camel_and_snakeCase"),
            vec!["strange", "mix", "of", "camel", "and", "snake", "case"]
        );
        assert_eq!(
            split_identifier("ignore123$_non-alphabetic"),
            vec!["ignore", "non", "alphabetic"]
        )
    }
}
