from functools import partial

from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block
from .diffloss import DiffLoss
from .mar import PointMAR

from huggingface_hub import PyTorchModelHubMixin
from pathlib import Path

import scipy.stats as stats
import math
import torch
import numpy as np
import torch.nn as nn


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class RasterPointMAR(PointMAR):
    def __init__(self, **kwargs):
        assert 'raster' not in kwargs, "raster should not be passed as a keyword argument"
        super().__init__(raster=True, **kwargs)
        

def raster_mar_pico(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=128, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=3,
        diffloss_w=128,
        **kwargs
    )
    return model

def raster_mar_nano(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=5,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=5,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=3,
        diffloss_w=256,
        **kwargs
    )
    return model

def raster_mar_tiny(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=4,
        diffloss_w=512,
        **kwargs
    )
    return model

def raster_mar_small(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=4,
        diffloss_w=768,
        **kwargs
    )
    return model

def raster_mar_base(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=6,
        diffloss_w=1024,
        **kwargs
    )
    return model


def raster_mar_large(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=8,
        diffloss_w=1280,
        **kwargs
    )
    return model


def raster_mar_huge(**kwargs):
    model = RasterPointMAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=12,
        diffloss_w=1536,
        **kwargs
    )
    return model


class RasterPointMARPipeline(
    RasterPointMAR,
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


def raster_mar_pico_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=128, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=3,
        diffloss_w=128,
        **kwargs
    )
    return model

def raster_mar_nano_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=5,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=5,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=3,
        diffloss_w=256,
        **kwargs
    )
    return model

def raster_mar_tiny_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=4,
        diffloss_w=512,
        **kwargs
    )
    return model

def raster_mar_small_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=4,
        diffloss_w=768,
        **kwargs
    )
    return model

def raster_mar_base_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=6,
        diffloss_w=1024,
        **kwargs
    )
    return model


def raster_mar_large_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=8,
        diffloss_w=1280,
        **kwargs
    )
    return model


def raster_mar_huge_pipeline(**kwargs):
    model = RasterPointMARPipeline(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        diffloss_d=12,
        diffloss_w=1536,
        **kwargs
    )
    return model