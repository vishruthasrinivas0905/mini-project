import cv2
import os

# Ask for student name
name = input("Enter student name: ")

# Create folder for that student
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

print(f"[INFO] Starting image capture for {name}")
print("[INFO] Press 'q' to stop early")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not access webcam.")
        break

    # Draw rectangle + counter
    cv2.putText(frame, f"Capturing {name} - Image {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Capture Window", frame)

    # Save every 5th frame
    if count % 5 == 0:
        img_path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[SAVED] {img_path}")

    count += 1

    # Stop after 100 frames or 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Captured {count//5} images for {name}")
