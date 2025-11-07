import cv2
import face_recognition
import pickle

# ✅ Path to your trained encodings file
ENCODINGS_PATH = "encodings.pkl"

# Load the trained encodings
print("[INFO] Loading known faces...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    recognized_names = set()
    frame_count = 0

    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        boxes = face_recognition.face_locations(rgb_frame, model="hog")
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        # Compare each detected face
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Pick the most frequent match
                name = max(counts, key=counts.get)

            if name != "Unknown":
                recognized_names.add(name)

    cap.release()
    print("[INFO] Finished processing video.")
    return recognized_names

if __name__ == "__main__":
    video_path = "test_video.mp4"  # 🎥 change to your actual video file name
    names = process_video(video_path)
    print("\n✅ Students recognized:")
    print(names)
