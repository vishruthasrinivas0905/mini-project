import os
import cv2
import face_recognition
import pickle

dataset_path = "dataset"
encoding_file = "encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Training started...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}...")

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print("[INFO] Serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}

with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Training complete! Encodings saved to encodings.pkl ✅")
