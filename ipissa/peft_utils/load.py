
from peft import PeftModel
from pathlib import Path
import safetensors.torch
from loguru import logger

def add_adapter_name_to_sd(sd, adapter_name="default", prefix="ipissa_"):
    new_sd = {}
    for k, v in sd.items():
        if prefix in k:
            new_k = f"{k}.{adapter_name}"
        new_sd[new_k] = v
    return new_sd


def remove_adapter_name(key, adapter_name="default"):
    if "." not in key:
        return key
    if key.endswith(f".{adapter_name}"):
        return key.removesuffix(f".{adapter_name}")
    return key  # .replace(f".{adapter_name}.", ".")




def save_adapter(model: PeftModel, save_folder: Path, adapter_name: str):
    """Save adapter weights and config."""

    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

    save_folder.mkdir(parents=True, exist_ok=True)

    config = model.peft_config[adapter_name]
    state_dict = model.state_dict()

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
    to_return = {k: state_dict[k] for k in state_dict if prefix in k}

    to_return = {remove_adapter_name(k, adapter_name): v for k, v in to_return.items()}

    safetensors.torch.save_file(to_return, save_folder / "adapter_model.safetensors")
    config.save_pretrained(save_folder)

    logger.info(f"Saved adapter to {save_folder}")
