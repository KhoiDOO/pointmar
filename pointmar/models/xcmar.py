from .xmar import XPointMAR

from huggingface_hub import PyTorchModelHubMixin
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn


class XCPointMAR(XPointMAR):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on XCPointMAR'
        super().__init__(causal=True, **kwargs)
    
    def sample_orders(self, bsz):
        order = np.arange(self.seq_len)
        orders = np.tile(order, (bsz, 1))
        orders = torch.tensor(orders, dtype=torch.long).cuda()
        return orders


def xcmar_pico(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=128, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, 
        diffloss_d=3,
        diffloss_w=128,
        **kwargs
    )
    return model

def xcmar_nano(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=5,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=5,
        mlp_ratio=4, 
        diffloss_d=3,
        diffloss_w=256,
        **kwargs
    )
    return model

def xcmar_tiny(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, 
        diffloss_d=4,
        diffloss_w=512,
        **kwargs
    )
    return model

def xcmar_small(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, 
        diffloss_d=4,
        diffloss_w=768,
        **kwargs
    )
    return model

def xcmar_base(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, 
        diffloss_d=6,
        diffloss_w=1024,
        **kwargs
    )
    return model


def xcmar_large(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, 
        diffloss_d=8,
        diffloss_w=1280,
        **kwargs
    )
    return model


def xcmar_huge(**kwargs):
    model = XCPointMAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, 
        diffloss_d=12,
        diffloss_w=1536,
        **kwargs
    )
    return model


class XCPointMARPipeline(
    XCPointMAR,
    PyTorchModelHubMixin,
    repo_url="https://github.com/KhoiDOO/pointmar",
    docs_url="https://github.com/KhoiDOO/pointmar",
    pipeline_tag="image-to-image",
    license="mit"
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load(self, path: str):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu", weights_only=False)

        self.load_state_dict(pkg['model'])


def xcmar_pico_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=128, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, 
        diffloss_d=3,
        diffloss_w=128,
        **kwargs
    )
    return model

def xcmar_nano_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=5,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=5,
        mlp_ratio=4, 
        diffloss_d=3,
        diffloss_w=256,
        **kwargs
    )
    return model

def xcmar_tiny_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, 
        diffloss_d=4,
        diffloss_w=512,
        **kwargs
    )
    return model

def xcmar_small_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, 
        diffloss_d=4,
        diffloss_w=768,
        **kwargs
    )
    return model

def xcmar_base_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, 
        diffloss_d=6,
        diffloss_w=1024,
        **kwargs
    )
    return model


def xcmar_large_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, 
        diffloss_d=8,
        diffloss_w=1280,
        **kwargs
    )
    return model


def xcmar_huge_pipeline(**kwargs):
    model = XCPointMARPipeline(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, 
        diffloss_d=12,
        diffloss_w=1536,
        **kwargs
    )
    return model