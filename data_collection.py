import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard


def palm_in_frame(results):
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    # Check if either left or right hand is detected
    if left_hand_landmarks or right_hand_landmarks:
        return True

    return False


file_path = "word_list.txt"

words_list = []
with open(file_path, "r") as file:
    for line in file:
        word = line.strip()
        words_list.append(word)

actions = np.array(words_list)

PATH = os.path.join("data")

sequences = 0  # Initialize sequence counter

for action in actions:
    sequence = 0  # Initialize sequence counter
    video_folder = os.path.join("videos", action)
    for _, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                sequences += 1  # Increment sequence counter
                sequence += 1  # Increment sequence counter
                video_path = os.path.join(video_folder, file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("Cannot open video:", video_path)
                    continue

                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get number of frames

                # palm_detected = False  # Flag variable for palm detection

                frameRenamed = 0

                for frame in range(frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Set frame position
                    ret, image = cap.read()
                    if not ret:
                        break

                    with mp.solutions.holistic.Holistic(
                        min_detection_confidence=0.75, min_tracking_confidence=0.75
                    ) as holistic:
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)

                    cv2.putText(
                        image,
                        'Recording data for the "{}". Sequence number {}.'.format(
                            action, sequence
                        ),
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Video", image)
                    cv2.waitKey(1)

                    if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                        break

                    # Check if the palm is detected
                    if palm_in_frame(results):
                        keypoints = keypoint_extraction(results)
                        frame_folder = os.path.join(PATH, action, str(sequence))
                        os.makedirs(
                            frame_folder, exist_ok=True
                        )  # Create frame folder if it doesn't exist
                        frame_path = os.path.join(frame_folder, str(frameRenamed) + ".npy")
                        np.save(frame_path, keypoints)
                        frameRenamed += 1

                    # save code was here

                cap.release()
                cv2.destroyAllWindows
