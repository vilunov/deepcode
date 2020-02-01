from deepcode.config import *


def test_parse():
    input = """
[languages.go]
train_path = "path1"
valid_path = "path2"
"""
    parsed = parse_config(input)
    expected = Config(
        languages={"go": Language(train_path="path1", valid_path="path2")}
    )

    assert parsed == expected
