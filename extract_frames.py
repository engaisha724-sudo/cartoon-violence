import cv2
import os

# Settings
DATASET_PATH = "D:/cartoon_detection/dataset"
OUTPUT_PATH = "D:/cartoon_detection/dataset/frames"
CLASSES = ["gun", "knife", "non_violent"]
FRAMES_PER_SECOND = 2

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        fps = 25
    
    frame_interval = max(1, int(fps / FRAMES_PER_SECOND))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count

def process_dataset():
    total = 0
    for label in CLASSES:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(label_path):
            print(f"No Folder : {label_path}")
            continue
        
        videos = [f for f in os.listdir(label_path) 
                  if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
        
        print(f"\n[{label.upper()}] — {len(videos)} videos")
        
        for video_file in videos:
            video_path = os.path.join(label_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(OUTPUT_PATH, label, video_name)
            
            count = extract_frames(video_path, output_folder)
            print(f"   {video_file[:40]}  {count} frames")
            total += count
    
    print(f"\nEnd !  Total frames: {total}")

if __name__ == "__main__":
    process_dataset()