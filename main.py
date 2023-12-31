import keyboard
import numpy as np
import os
import mediapipe as mp
import cv2
from my_functions import *

import tensorflow
from tensorflow import keras

from keras.models import load_model
from keras.utils import pad_sequences

PATH = os.path.join("data")

actions = np.array(os.listdir(PATH))

model = load_model("my_model")

sentence, keypoints = [" "], []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access the camera.")
    exit()

with mp.solutions.holistic.Holistic(
    min_detection_confidence=0.75, min_tracking_confidence=0.75
) as holistic:
    while cap.isOpened():
        _, image = cap.read()
        results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 54:
            keypoints = np.array(keypoints)
            keypoints = pad_sequences([keypoints], maxlen=54, dtype="float32")
            prediction = model.predict(keypoints)
            keypoints = []

            print(np.amax(prediction))

            if np.amax(prediction) > 0.5:
                if sentence[-1] != actions[np.argmax(prediction)]:
                    sentence.append(actions[np.argmax(prediction)])

        if len(sentence) > 7:
            sentence = sentence[-7:]

        if keyboard.is_pressed(" "):
            sentence = [" "]

        textsize = cv2.getTextSize(" ".join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[
            0
        ]
        text_X_coord = (image.shape[1] - textsize[0]) // 2

        cv2.putText(
            image,
            " ".join(sentence),
            (text_X_coord, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        print(sentence)

        cv2.imshow("Camera", image)

        cv2.waitKey(1)
        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
