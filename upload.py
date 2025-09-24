from models import mar
from models.mar import PointMARPipeline
from util.config import load_args_from_json

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Point-MAR inference script')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    args = parser.parse_args()

    output_dir = args.output_dir
    args = load_args_from_json(os.path.join(args.output_dir, 'config.json'))

    pipeline: PointMARPipeline = mar.__dict__[f"{args.model}_pipeline"](
        num_points=args.num_points,
        token_embed_dim=args.token_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    n_params = sum(p.numel() for p in pipeline.parameters())
    print("Number of parameters: {}M".format(n_params / 1e6))

    best_checkpoint = os.path.join(output_dir, 'checkpoint-best.pth')
    last_checkpoint = os.path.join(output_dir, 'checkpoint-last.pth')
    print(last_checkpoint, os.path.exists(last_checkpoint))
    if os.path.exists(best_checkpoint):
        checkpoint_path = best_checkpoint
    elif os.path.exists(last_checkpoint):
        checkpoint_path = last_checkpoint
    else:
        raise FileNotFoundError("No checkpoint found in the output directory.")

    pipeline.load(checkpoint_path)

    repo_id = f"{args.dataset_name}_{args.model}_{args.num_points}pts_{args.token_embed_dim}dim_{args.diffloss_d}_{args.diffloss_w}"

    try:
        pipeline.push_to_hub(repo_id=repo_id)
    except Exception as e:
        print(f"Failed to push model to Hugging Face: {e}")