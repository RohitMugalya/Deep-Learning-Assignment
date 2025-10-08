import cv2
import os
import subprocess
from natsort import natsorted


def process_video(video_path):
    # Step 1: Extract frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join("video_frames", video_name)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count}.jpg"), frame)
        frame_count += 1
    cap.release()
    print(f"✅ Extracted {frame_count} frames to '{frames_dir}'")

    # Step 2: Run test.py subprocess
    enhanced_frames_dir = os.path.join("low_light_video_enhanced", "frames", video_name)
    os.makedirs(enhanced_frames_dir, exist_ok=True)

    cmd = [
        "python", "test.py",
        "--input_dir", frames_dir,
        "--weight_dir", "weight/Epoch99.pth",
        "--test_dir", enhanced_frames_dir
    ]
    print("🚀 Running enhancement subprocess...")
    subprocess.run(cmd, check=True)
    print("✅ Enhancement complete!")

    # Step 3: Stitch enhanced frames into video
    output_video_dir = os.path.join("low_light_video_enhanced", "videos")
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{video_name}_enhanced.mp4")

    # Collect all enhanced frames
    frame_files = natsorted([
        os.path.join(enhanced_frames_dir, f)
        for f in os.listdir(enhanced_frames_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    if not frame_files:
        print(f"❌ No enhanced frames found in {enhanced_frames_dir}")
        return

    # Read first frame to get size
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fps = 30  # You can adjust based on input video if needed

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)
    out.release()

    print(f"🎬 Video saved at: {output_video_path}")

# Example usage:
process_video("video_inputs/sample1.mp4")
