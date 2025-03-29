import cv2  # This helps with images & videos
import os   # This helps with folders & files

#  Where the videos are stored
video_folder = "/path/to/Le2i/Falls/"  
frame_output = "/path/to/Le2i/Frames/Falls/"

#  Make a folder to save the pictures
os.makedirs(frame_output, exist_ok=True)

#  Go through every video in the folder
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)  #  Open the video
    count = 0  # Start counting frames

    while cap.isOpened():
        ret, frame = cap.read()  #  Take one picture (frame)
        if not ret:
            break  #  Stop when there are no more pictures

        # Save the picture in the folder
        frame_path = os.path.join(frame_output, f"{video_file}_frame{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1  # Move to the next frame

    cap.release()  # 🔚 Close the video

print("All frames saved! 🎉")
->3rd
