from tokenizers import ByteLevelBPETokenizer


tokenizer_doc = ByteLevelBPETokenizer("../cache/vocabs/doc-vocab.json", "../cache/vocabs/doc-merges.txt")
tokenizers_code = {
    code: ByteLevelBPETokenizer(
        f"../cache/vocabs/code-{code}-vocab.json", f"../cache/vocabs/code-{code}-merges.txt"
    )
    for code in ["java", "javascript", "php", "ruby", "python", "go"]
}

