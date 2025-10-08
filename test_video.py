import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test Video Enhancement Pipeline")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Step 1: Extract frames (V2I)
    v2i_cmd = [
        "python", "make_video.py",
        "--video_path", video_path,
        "--image_lowlight_folder", f"video_frames/{video_name}/%d.jpg",
        "--choice", "V2I"
    ]
    print(f"🚀 [1/2] Extracting frames from {video_path}")
    subprocess.run(v2i_cmd, check=True)
    print("✅ Frame extraction complete!")

    # Step 2: Run enhancement model (test.py)
    print("⚙️ [Intermediate] Running test.py for enhancement...")
    enhance_cmd = [
        "python", "test.py",
        "--input_dir", f"video_frames/{video_name}/",
        "--weight_dir", "weight/Epoch99.pth",
        "--test_dir", f"low_light_video_enhanced/frames/{video_name}/"
    ]
    subprocess.run(enhance_cmd, check=True)
    print("✅ Enhancement complete!")

    # Step 3: Combine enhanced frames into video (I2V)
    i2v_cmd = [
        "python", "make_video.py",
        "--video_path", video_path,
        "--image_folder", f"low_light_video_enhanced/frames/{video_name}/",
        "--save_path", f"low_light_video_enhanced/videos/{video_name}_enhanced.mp4",
        "--choice", "I2V"
    ]
    print("🎬 [2/2] Stitching enhanced frames into final video...")
    subprocess.run(i2v_cmd, check=True)
    print(f"✅ Done! Enhanced video saved at: low_light_video_enhanced/videos/{video_name}_enhanced.mp4")

if __name__ == "__main__":
    main()
