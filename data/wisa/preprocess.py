import os
import json
import cv2
import argparse
import shutil
import zipfile

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")
        return False


def collect_videos(root_dir):
    video_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                video_paths.append(os.path.join(root, f))
    return video_paths


def flatten_videos(root_dir):
    video_paths = collect_videos(root_dir)

    for vp in video_paths:
        # Move everything to the top level of root_dir
        dst = os.path.join(root_dir, os.path.basename(vp))

        if os.path.abspath(vp) == os.path.abspath(dst):
            continue

        base, ext = os.path.splitext(dst)
        i = 1
        while os.path.exists(dst):
            dst = f"{base}_{i}{ext}"
            i += 1

        shutil.move(vp, dst)


def remove_empty_dirs(root_dir):
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for d in dirs:
            full = os.path.join(root, d)
            if not os.listdir(full):
                os.rmdir(full)


def preprocess_folders(folder):
    folder = os.path.abspath(folder)

    # 1. unzip all zip files
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".zip"):
                continue
            zip_path = os.path.join(root, f)
            if extract_zip(zip_path, root):
                os.remove(zip_path)

    # 2. move all videos to root
    flatten_videos(folder)

    # 3. clean empty folders
    remove_empty_dirs(folder)


def remove_short_videos(folder_path, min_frames=81, extensions=VIDEO_EXTS):
    folder_path = os.path.abspath(folder_path)
    removed_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(extensions):
                continue

            video_path = os.path.join(root, filename)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {video_path}, skipping.")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count < min_frames:
                os.remove(video_path)
                removed_count += 1
    
    return removed_count


def generate_txt(video_folder, json_path, output_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_dict = {item["video_name"]: item for item in data}

    captions_list = []
    video_names_list = []
    matched_count = 0

    for root, _, files in os.walk(video_folder):
        folder_name = os.path.basename(root)

        for file_name in files:
            if file_name in video_dict:
                video_path_str = f"{folder_name}/{file_name}"
                video_names_list.append(video_path_str)

                caption = video_dict[file_name]["captions"]
                caption = caption.replace("\n", " ").replace("\r", " ").strip()
                captions_list.append(caption if caption else "no description")
                matched_count += 1

    os.makedirs(output_dir, exist_ok=True)

    captions_path = os.path.join(output_dir, "prompt.txt")
    video_names_path = os.path.join(output_dir, "video.txt")

    with open(captions_path, "w", encoding="utf-8") as f:
        for c in captions_list:
            f.write(c + "\n")

    with open(video_names_path, "w", encoding="utf-8") as f:
        for v in video_names_list:
            f.write(v + "\n")
            
    return matched_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A unified script to process videos and generate annotations.")

    parser.add_argument("--video_folder", type=str, default="./videos", help="Directory containing videos/zips")
    parser.add_argument("--json_path", type=str, default="./wisa-80k.json", help="Path to metadata JSON")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save prompt.txt and video.txt")
    parser.add_argument("--min_frames", type=int, default=81, help="Minimum frame count to keep a video")

    args = parser.parse_args()

    print(f"[*] Step 1: Extracting zips, flattening video paths, and cleaning empty folders in '{args.video_folder}'...")
    preprocess_folders(args.video_folder)

    print(f"[*] Step 2: Removing videos with fewer than {args.min_frames} frames...")
    removed = remove_short_videos(args.video_folder, args.min_frames)
    print(f"    -> Removed {removed} short videos.")

    print(f"[*] Step 3: Generating prompt.txt and video.txt based on '{args.json_path}'...")
    matched = generate_txt(args.video_folder, args.json_path, args.output_dir)
    print(f"    -> Successfully matched and wrote {matched} videos to txt files.")

    print("\n[+] All preprocessing completed successfully!")