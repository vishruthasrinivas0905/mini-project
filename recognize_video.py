import cv2
import face_recognition
import pickle
import json
import smtplib
import sqlite3
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from email.message import EmailMessage
from datetime import datetime

# --- CONFIGURATION ---
ENCODINGS_PATH = "encodings.pkl"
TIMETABLE_PATH = "timetable.json"
DB_PATH = "attendance_db.sqlite"

SENDER_EMAIL = "vishrutha.srinivas@gmail.com"
SENDER_PASSWORD = "pruk kfjb aajl arkf"  
RECEIVER_EMAILS = ["1by23ai188@bmsit.in", "1by23ai181@bmsit.in", "1by23ai173@bmsit.in"]

def get_current_slot():
    """Objective 2: Mapping Classification to Class and Time"""
    now = datetime.now()
    current_day = now.strftime("%A") 
    current_time = now.strftime("%H:%M")
    try:
        with open(TIMETABLE_PATH, "r") as f:
            schedule = json.load(f)
        if current_day in schedule:
            for slot in schedule[current_day]:
                if slot["start"] <= current_time <= slot["end"]:
                    return slot["subject"], f"{slot['start']} - {slot['end']}"
        return "Special Session", "N/A"
    except:
        return "General Class", "N/A"

def send_final_confirmation_email(subject, timing, students_count, intruder_found):
    """Objective 3: Formal Alert after Confirmation"""
    if not intruder_found:
        print("✅ Analysis complete: No unauthorized individuals confirmed.")
        return

    print("📧 Sending Final Security Confirmation Email...")
    try:
        msg = EmailMessage()
        msg['Subject'] = f"SECURITY BREACH CONFIRMED: {subject}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(RECEIVER_EMAILS)
        
        msg.set_content(f"""
Dear Administrator,

The AI Facial Recognition System has completed the final classification of the video feed.

--- SESSION SUMMARY ---
Subject: {subject}
Timing: {timing}
Date: {datetime.now().strftime('%B %d, %Y')}
Students Present: {students_count}
Security Status: INTRUDER CONFIRMED

An unauthorized individual was detected during this session. The logs have been updated with the classification data.

Regards,
AI Security Division
BMS Institute of Technology
""")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email notification sent to administrators.")
    except Exception as e:
        print(f"❌ Email Error: {e}")

def process_attendance():
    root = tk.Tk()
    root.withdraw()
    source = filedialog.askopenfilename(title="Select Video for Analysis")
    if not source: return

    # Database Initialization
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS attendance") 
    cursor.execute("CREATE TABLE attendance (name TEXT, subject TEXT, timing TEXT, date TEXT, timestamp TEXT)")
    conn.commit()

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(source)
    recognized_names = set()
    intruder_frame_count = 0
    
    # Get Class and Time Mapping
    subject, timing = get_current_slot()

    print(f"🚀 CLASSIFYING: Starting analysis for {subject} ({timing})...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Frame Skipping for Performance
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 3 != 0: continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
        
        unknown_detected = False

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            # OBJECTIVE 1: Accurate Classification
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]

            # Threshold 0.50 (Recognizes students accurately)
            if min_distance < 0.44:
                name = data["names"][best_match_index]
                conf = f"{round((1 - min_distance) * 100)}%"
            else:
                name = "Unknown"
                unknown_detected = True
                conf = ""

            # OBJECTIVE 2: Save Classification with Class and Time
            if name != "Unknown" and name not in recognized_names:
                now = datetime.now()
                cursor.execute("INSERT INTO attendance VALUES (?, ?, ?, ?, ?)", 
                             (name, subject, timing, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
                conn.commit()
                recognized_names.add(name)
                print(f"✅ Classified: {name} | Time: {now.strftime('%H:%M:%S')}")

            # UI Display
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} {conf}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if unknown_detected:
            intruder_frame_count += 1

        cv2.imshow("BMSIT AI Classification System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- FINAL STEP ---
    cap.release()
    cv2.destroyAllWindows()
    
    # OBJECTIVE 3: Confirm Intruder and Send Email
    # Only confirms if intruder was seen for more than 10 frames
    intruder_confirmed = True if intruder_frame_count > 10 else False
    send_final_confirmation_email(subject, timing, len(recognized_names), intruder_confirmed)

    conn.close()
    messagebox.showinfo("Analysis Complete", f"Classified {len(recognized_names)} students.\nReport processed for {subject}.")

if __name__ == "__main__":
    process_attendance()