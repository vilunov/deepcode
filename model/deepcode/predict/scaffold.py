from deepcode.config import Config
from deepcode.scaffold import AbstractScaffold

from tokenizers import ByteLevelBPETokenizer


class PredictScaffold(AbstractScaffold):
    def __init__(self, config: Config, weights_path: str):
        super().__init__(config, weights_path)
        self.tokenizer_doc = ByteLevelBPETokenizer("../cache/vocabs/doc-vocab.json", "../cache/vocabs/doc-merges.txt")
        self.tokenizers_code = {
            code: ByteLevelBPETokenizer(f"../cache/vocabs/code-{code}-vocab.json", f"../cache/vocabs/code-{code}-merges.txt")
            for code in config.model.code_encoder.keys()
        }
