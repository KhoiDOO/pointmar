import os
import argparse
import torch
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from pointmar.models.mar import PointMARPipeline


def export_to_ply(points: np.ndarray, filename: str):
    """
    Exports a NumPy array of shape (N, 3) to a PLY file using Open3D.
    
    :param points: NumPy array of shape (N, 3)
    :param filename: Output file path (.ply)
    """
    print(f"\n--- Starting PLY Export ---")
    
    # 1. Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 2. Add colors
    # Using a uniform blue color for all points in this example
    colors = np.array([[0.1, 0.4, 0.8]] * points.shape[0], dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3. Write to file
    try:
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        print(f"✅ Successfully exported {points.shape[0]} points to '{filename}'")
    except Exception as e:
        print(f"❌ Failed to write PLY file: {e}")


def render_360_video_and_views(points: np.ndarray, output_dir: str, video_file: str, steps: int, elev: int, fps: int):
    """
    Renders frames of the point cloud rotating 360 degrees, saves them as PNG 
    files, and stitches them into a video using Matplotlib and OpenCV.
    
    :param points: NumPy array of shape (N, 3)
    :param output_dir: Directory to save the temporary PNG frames.
    :param video_file: The final output video file path (.mp4 recommended).
    :param steps: Number of rotation steps (360 / steps = degrees per step).
    :param elev: The fixed elevation angle (vertical camera tilt).
    :param fps: Frames per second for the output video.
    """
    print(f"\n--- Starting 360 Video and Frame Rendering ({steps} views) ---")
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    frame_files = [] # To store paths of generated PNGs
    
    # Determine plot limits for a consistent view
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5

    # 1. Create Figure and 3D Axes (Using Agg backend for headless systems)
    # Set the backend to 'Agg' to ensure rendering works without a display server
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. Scatter Plot the Points
    # Color points based on Z height 
    colors = Z
    ax.scatter(X, Y, Z, c=colors, cmap='viridis', marker='o', s=10)
    
    # 3. Configure Plot Aesthetics
    ax.set_title("360 Point Cloud View")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set equal aspect ratio for correct shape representation
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Remove grid and axis ticks for a cleaner video background
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Calculate step size
    angle_step = 360.0 / steps
    
    # 4. Rotation and Capture Loop (PNG frames)
    print(f"Generating PNG frames in '{output_dir}/'...")
    for i in range(steps):
        angle = i * angle_step
        
        # Set the camera view: elev (fixed), azim (rotating)
        ax.view_init(elev=elev, azim=angle)
        
        # Save the figure as a high-quality PNG (suitable for video encoding)
        filename = os.path.join(output_dir, f"frame_{i:04d}_azim{int(angle):03d}deg.png")
        fig.savefig(filename, format='png', bbox_inches='tight')
        frame_files.append(filename)
        
        if (i + 1) % 10 == 0:
            print(f"Captured {i + 1}/{steps} frames...")

    plt.close(fig)
    print(f"✅ Finished capturing {steps} PNG frames.")
    
    # 5. Video Stitching using OpenCV
    print(f"Starting video stitching to '{video_file}' using OpenCV...")
    
    try:
        # Read the first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            raise FileNotFoundError("Could not read the first frame image. Check file path/permissions.")
            
        height, width, layers = first_frame.shape
        
        # Define the codec (e.g., 'mp4v' for MP4) and create VideoWriter object
        # Note: If 'mp4v' fails, try 'XVID' for AVI, or ensure required codecs are installed.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
        
        # Write all frames to the video file
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            # VideoWriter requires BGR frames, which cv2.imread provides
            video.write(frame) 
            # Optionally uncomment to clean up the intermediate PNGs after video creation
            # os.remove(frame_file) 

        video.release()
        print(f"✅ Successfully created video: '{video_file}' (FPS: {fps})")

    except Exception as e:
        print(f"❌ Failed to create video using OpenCV. Error: {e}")
        print("Suggestion: Ensure 'opencv-python' is installed and the video codec ('mp4v') is supported on your system.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mar_base', help='Name of the model to use')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to use')
    parser.add_argument('--token_embed_dim', type=int, default=3, help='Token embedding dimension')
    parser.add_argument('--num_samples', type=int, default=27, help='Number of samples to generate')
    parser.add_argument('--num_ar_steps', type=int, default=256, help='Number of autoregressive steps')
    parser.add_argument('--cfg_schedule', type=str, default="constant", help='CFG schedule type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--export_ply', action='store_true', help='Whether to export samples to PLY files')
    parser.add_argument('--export_video', action='store_true', help='Whether to export sampling process as a video')

    args = parser.parse_args()

    assert args.export_ply or args.export_video, "At least one of --export_ply or --export_video must be set."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    key = f"shapenet_{args.model_name}_{args.num_points}pts_{args.token_embed_dim}dim"
    model_key = f"kohido/{key}"
    model = PointMARPipeline.from_pretrained(model_key, cache_dir="./.cache")
    model.to(device)
    model.eval()

    with torch.amp.autocast(device_type=device):
        sampled_tokens = model.sample_tokens(
            bsz=args.num_samples,
            num_iter=args.num_ar_steps,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            progress=True
        )
    
    point_clouds = sampled_tokens.cpu().numpy()

    # Export each point cloud to a PLY file
    output_dir = f"./docs/samples/{key}"
    if args.export_ply:
        ply_output_dir = os.path.join(output_dir, "ply")

        os.makedirs(ply_output_dir, exist_ok=True)

        for i, pc in enumerate(point_clouds):
            ply_filename = os.path.join(ply_output_dir, f"sample_{i}.ply")
            export_to_ply(pc, ply_filename)
        
        print(f"\nAll samples have been exported to '{ply_output_dir}'")

    # Optionally, export the sampling process as a video
    if args.export_video:
        video_output_dir = os.path.join(output_dir, "video")
        os.makedirs(video_output_dir, exist_ok=True)

        for i, pc in enumerate(point_clouds):
            video_frames_dir = os.path.join(video_output_dir, f"sample_{i}_frames")
            os.makedirs(video_frames_dir, exist_ok=True)
            video_filename = os.path.join(video_output_dir, f"sample_{i}_360view.mp4")

            render_360_video_and_views(
                points=pc,
                output_dir=video_frames_dir,
                video_file=video_filename,
                steps=120,      # 120 frames for a smooth 360 rotation
                elev=30,       # Fixed elevation angle
                fps=30         # 30 FPS video
            )
        
        print(f"\nAll videos have been exported to '{video_output_dir}'")