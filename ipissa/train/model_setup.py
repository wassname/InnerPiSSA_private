"""Model initialization and setup utilities for InnerPiSSA training."""

from typing import Dict, List, Optional

import torch
from baukit import TraceDict
from loguru import logger
from peft import PeftModel
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

from ipissa.config import TrainingConfig
from ipissa.peft_utils.innerpissa import InnerPiSSAConfig, register_ipissa_peft


def load_model(model_id, quantization_type="none"):
    """Load base model with optional quantization."""
    model_kwargs = {}
    if quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs['quantization_config'] = quantization_config
    elif quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs['quantization_config'] = quantization_config

    logger.info(f"Loading model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda:0",
        **model_kwargs
    )

    if 'quantization_config' in model_kwargs:
        base_model.enable_input_require_grads()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    return base_model, tokenizer


def setup_adapter(base_model, config: TrainingConfig, target_modules: str, init_steering_vecs=None):
    """Setup InnerPiSSA adapter on base model.
    
    Args:
        base_model: Base model to add adapter to
        config: Training configuration
        target_modules: PEFT target_modules regex (from LayerSelection)
        init_steering_vecs: Optional dict of {layer_name: dHS_tensor} for data-aware init
    """
    logger.info(f"Target modules regex: {target_modules}")

    if config.adapter_type == "innerpissa":
        adapter_config = InnerPiSSAConfig(
            r=config.r,
            scale_s=config.scale_s,
            rotate_u=config.rot_u,
            rotate_v=config.rot_v,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            steering_vectors=init_steering_vecs,
        )
    else:  # lora or dora
        from peft import LoraConfig
        adapter_config = LoraConfig(
            r=config.r,
            lora_alpha=config.r,  # Common default: alpha=r
            lora_dropout=0.0,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            use_dora=(config.adapter_type == "dora"),
        )

    model = PeftModel(base_model, adapter_config, adapter_name=config.dataset_name)
    logger.info(
        f"Adapter configured: type={config.adapter_type}, rank={config.r}, target_modules={target_modules}"
    )

    return model


def extract_U_matrices(model, loss_layers: List[str], config: TrainingConfig):
    """Extract SVD U, S, V matrices for weight reconstruction.
    
    Works with both adapter layers (extracts from base_layer.weight) and 
    non-adapter layers (extracts from weight directly).
    """
    Uw_full = {}
    Vw_full = {}
    Sw_full = {}

    for lk in tqdm(loss_layers, desc='svd'):
        m = model.get_submodule(lk)
        
        # Check if this is an adapter layer (has base_layer) or regular layer
        if hasattr(m, 'base_layer'):
            W = m.base_layer.weight.data.float()
        else:
            W = m.weight.data.float()
        
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Uw_full[lk] = U.to(model.device).float()
        Sw_full[lk] = S.to(model.device).float()
        Vw_full[lk] = Vh.T.to(model.device).float()  # V = Vh.T

    shapes = {k: v.shape for k, v in Uw_full.items()}
    logger.info(f"Extracted U matrices: {shapes}")

    return Uw_full, Sw_full, Vw_full


def compute_init_steering_vectors(
    model, dataset_pt, loss_layers, tokenizer, config, n_samples=32
):
    """Compute raw dHS from first batch for data-aware adapter initialization.
    
    Args:
        model: Base model (no adapter yet)
        dataset_pt: Tokenized dataset
        loss_layers: Layers to extract activations from
        tokenizer: Tokenizer
        config: TrainingConfig
        n_samples: Number of samples to use (must be even for cho/rej pairs)
    
    Returns:
        Dict[layer_name, dHS_tensor] where dHS = mean(hs_cho - hs_rej)
    """
    # Take first n_samples (must be even for pairs)
    assert n_samples % 2 == 0, "n_samples must be even for cho/rej pairs"
    subset_indices = list(range(min(n_samples, len(dataset_pt))))
    subset = Subset(dataset_pt, subset_indices)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(subset, batch_size=config.bs, collate_fn=data_collator)
    
    model.eval()
    steering_vecs = {layer: [] for layer in loss_layers}
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            attention_mask = batch["attention_mask"]
            
            with TraceDict(model, layers=loss_layers) as ret:
                model(**batch)
            
            # Extract last token activations
            for layer in loss_layers:
                hs = (ret[layer].output * attention_mask.unsqueeze(-1)).float()
                hs_cho = hs[::2].mean(dim=1)  # [n_pairs, d] -> avg over seq
                hs_rej = hs[1::2].mean(dim=1)
                dHS = (hs_cho - hs_rej)  # [n_pairs, d] - keep per-pair
                steering_vecs[layer].append(dHS.cpu())
    
    # Concatenate across batches (keep per-pair)
    steering_vecs = {k: torch.cat(v, dim=0) for k, v in steering_vecs.items()}
    return steering_vecs
