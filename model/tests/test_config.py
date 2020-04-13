from deepcode.config import *


def test_parse():
    input = """
[training]
learning_rate = 0.1
data_train = "../cache/data/train.h5"
data_valid = "../cache/data/valid.h5"
loss_type = "crossentropy"
dropout_rate = 0.2
batch_size_train = 512
batch_size_valid = 100
func_name_share = 0.1
title = "test_config"

[model]
encoded_dims = 128

[model.doc_encoder]
type = "nbow"

[model.code_encoder.go]
type = "nbow"
"""
    parsed = parse_config(input)
    expected = Config(
        training=Training(
            learning_rate=0.1,
            data_train="../cache/data/train.h5",
            data_valid="../cache/data/valid.h5",
            loss_type="crossentropy",
            loss_margin=None,
            dropout_rate=0.2,
            batch_size_train=512,
            batch_size_valid=100,
            func_name_share=0.1,
            title="test_config",
        ),
        model=Model(
            encoded_dims=128,
            doc_encoder=Encoder(type="nbow", pooling_type=None),
            code_encoder={"go": Encoder(type="nbow", pooling_type=None)},
        ),
    )

    assert parsed == expected
