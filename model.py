import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

# Start Modification

data_folder = "data"
actions = os.listdir(data_folder)

sequences = []
frames = float("inf")  # Set to a large number initially

# Iterate through the data folder and collect sequence and frame numbers
for action in actions:
    action_folder = os.path.join(data_folder, action)
    if not os.path.isdir(action_folder):
        continue
    sequence_folders = os.listdir(action_folder)
    for sequence_folder in sequence_folders:
        sequence_folder_path = os.path.join(action_folder, sequence_folder)
        if not os.path.isdir(sequence_folder_path):
            continue
        frame_files = os.listdir(sequence_folder_path)
        num_frames = len(frame_files)
        if num_frames < frames:
            frames = num_frames
        sequences.append((action, sequence_folder, num_frames))

label_map = {label: num for num, label in enumerate(actions)}

landmarks, labels = [], []

for action, sequence_folder, num_frames in sequences:
    action_folder = os.path.join(data_folder, action)
    sequence_folder_path = os.path.join(action_folder, sequence_folder)
    temp = []
    for frame in range(num_frames):
        npy_path = os.path.join(sequence_folder_path, str(frame) + ".npy")
        npy = np.load(npy_path)
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# Pad sequences to have the same length
X = pad_sequences(landmarks, padding="post", dtype="float32")
Y = to_categorical(labels).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=34, stratify=Y
)

Y_train = np.array(Y_train)

model = Sequential()
model.add(
    LSTM(
        32,
        return_sequences=True,
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )
)
model.add(LSTM(64, return_sequences=True, activation="relu"))
model.add(LSTM(32, return_sequences=False, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(actions), activation="softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
model.fit(X_train, Y_train, epochs=100)

model.save("my_model")

predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)
