from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_config(cfg, model_name):
    for model_cfg in cfg["experiments"]["models"]:
        if model_cfg["name"] == model_name:
            return model_cfg

    raise ValueError(f"Unknown model: {model_name}")


def load_model(model_name, quantization="fp16", cfg=None):
    if cfg is None:
        cfg = load_config()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for model inference.")

    model_cfg = get_model_config(cfg, model_name)
    model_id = model_cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )

    elif quantization == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

    elif quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

    else:
        raise ValueError(
            f"Unknown quantization mode: '{quantization}'."
        )

    model.eval()

    return model, tokenizer