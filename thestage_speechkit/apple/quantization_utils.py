import mlx.core as mx
import mlx.nn as nn


def quantize_mlx_model(model, quant_config):
    def quantization_class_predicate(path, module):
        if path in quant_config:
            return quant_config[path]
        return False

    nn.quantize(model, class_predicate=quantization_class_predicate)

    return model
