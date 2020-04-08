use hdf5::H5Type;
use serde::Deserialize;

pub const MAX_LEN: usize = 1024;
pub const MAX_LEN2: usize = 256;
pub const MAX_NAME_LEN: usize = 32;

#[derive(Deserialize)]
pub struct SnippetCode {
    pub code_tokens: Vec<String>,
}

#[derive(Deserialize)]
pub struct SnippetDoc {
    pub docstring_tokens: Vec<String>,
    pub func_name: String,
}

#[derive(Deserialize)]
pub struct SnippetBoth {
    pub code_tokens: Vec<String>,
    pub docstring_tokens: Vec<String>,
    pub func_name: String,
}

#[derive(Deserialize)]
pub struct SnippetSearch {
    pub function_tokens: Vec<String>,
    pub url: String,
    pub identifier: String,
}

#[repr(C)]
#[derive(H5Type)]
pub struct Snippet {
    pub code_len: usize,
    pub code_tokens: [u32; MAX_LEN],
    pub doc_len: usize,
    pub doc_tokens: [u32; MAX_LEN],
    pub name_len: usize,
    pub name_tokens: [u32; MAX_NAME_LEN],
}

#[repr(C)]
#[derive(H5Type)]
pub struct Snippet2 {
    pub code_len: usize,
    pub code_tokens: [u32; MAX_LEN2],
}
