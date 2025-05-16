import cv2
from ultralytics import YOLO
from plyer import notification
import datetime
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Load models
model_person = YOLO('yolov8n.pt')  # Use YOLOv8 Nano for faster performance
model_fall = YOLO(r"C:\Users\susmi\Downloads\phase3&4initial\phase3&4initial\best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

# Optional: reduce buffer delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Create folder to save screenshots
save_dir = "fall_screenshots"
os.makedirs(save_dir, exist_ok=True)

# Email Configuration
sender_email = "susmithamanthena7@gmail.com"
receiver_email = "susmithamanthena7@gmail.com"
password = "bksp mlcl kfbi lvsy"  # Use app-specific password (never share real password)

def send_email(subject, body, to_email, attachment_path=None):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if attachment_path:
        with open(attachment_path, 'rb') as file:
            img = MIMEImage(file.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
            msg.attach(img)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Function to handle alert in a separate thread
def handle_alert(frame, filepath, timestamp):
    cv2.imwrite(filepath, frame)
    send_email(
        subject="⚠️ Fall Detected!",
        body=f"A fall was detected at {timestamp}. Please check the attached screenshot.",
        to_email=receiver_email,
        attachment_path=filepath
    )

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Step 1: Person detection
    results_person = model_person(resized_frame)
    person_detected = False

    for box in results_person[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0:  # Class 0 = person
            person_detected = True
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, 'Person', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 2: Fall detection only if person detected
    fall_detected = False
    if person_detected:
        results_fall = model_fall(resized_frame)

        for box in results_fall[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_fall.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if label == 'NonFalls' else (0, 0, 255)

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == 'Falls' and conf > 0.85:
                fall_detected = True

    if fall_detected:
        # Notification
        notification.notify(
            title="⚠️ Fall Detected!",
            message="Fall detected on live camera feed.",
            timeout=10
        )

        # Save screenshot & send email in background
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_screenshot_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.putText(resized_frame, "⚠️ FALL DETECTED ⚠️", (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        threading.Thread(target=handle_alert, args=(resized_frame.copy(), filepath, timestamp)).start()

    # Show result
    cv2.imshow('Live Fall Detection', resized_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
