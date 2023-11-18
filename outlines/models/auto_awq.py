from .transformers import Transformers
from typing import Optional

def auto_awq(
        model_name: str,
        fuse_layers: bool=True,
        device: Optional[str] = None,
        model_kwargs: dict={},
        tokenizer_kwargs: dict={}
    ):

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "The `auto_awq` and `transformers` library needs to be installed in order to use `AutoAWQ` models."
        )

    model_kwargs["fuse_layer"] = fuse_layer

    if device is not None:
        model_kwargs["device_map"] = device


    model = AutoAWQForCausalLM.from_quantized(quant_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

    return Transformers(model, tokenizer)
