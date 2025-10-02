from pointmar.models import mar
from pointmar.models.mar import PointMARPipeline
from pointmar.util.config import load_args_from_json

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Point-MAR inference script')
    parser.add_argument('--path', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--last', action='store_true', help='Use last checkpoint for inference.')
    args = parser.parse_args()

    exp_config = load_args_from_json(os.path.join(args.path, 'config.json'))

    pipeline: PointMARPipeline = mar.__dict__[f"{exp_config.model}_pipeline"](
        num_points=exp_config.num_points,
        token_embed_dim=exp_config.token_embed_dim,
        mask_ratio_min=exp_config.mask_ratio_min,
        attn_dropout=exp_config.attn_dropout,
        proj_dropout=exp_config.proj_dropout,
        buffer_size=exp_config.buffer_size,
        num_sampling_steps=exp_config.num_sampling_steps,
        diffusion_batch_mul=exp_config.diffusion_batch_mul,
        grad_checkpointing=exp_config.grad_checkpointing,
    )

    n_params = sum(p.numel() for p in pipeline.parameters())
    print("Number of parameters: {}M".format(n_params / 1e6))

    best_checkpoint = os.path.join(args.path, 'checkpoint-best.pth')
    last_checkpoint = os.path.join(args.path, 'checkpoint-last.pth')
    if args.last:
        checkpoint_path = last_checkpoint
    elif os.path.exists(best_checkpoint):
        checkpoint_path = best_checkpoint
    elif os.path.exists(last_checkpoint):
        checkpoint_path = last_checkpoint
    else:
        raise FileNotFoundError("No checkpoint found in the output directory.")

    pipeline.load(checkpoint_path)

    repo_id = f"{exp_config.dataset_name}_{exp_config.model}_{exp_config.num_points}pts_{exp_config.token_embed_dim}dim_{'last' if args.last else 'best'}"

    try:
        pipeline.push_to_hub(repo_id=repo_id)
    except Exception as e:
        print(f"Failed to push model to Hugging Face: {e}")