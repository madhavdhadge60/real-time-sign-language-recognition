import tensorflow as tf
import numpy as np
import os
import cv2
import json
import mediapipe as mp

# ================= PERFORMANCE =================
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================= INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# ================= CONFIG =================
DATASET_PATH = "md_asl_alphabet_train"   # My own created Dataset
IMG_SIZE = 160
BATCH_SIZE = 32
SEQ_LEN = 1

# ================= DATA =================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Class indices:", train_data.class_indices)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ================= HAND DETECTION PREPROCES =================
def detect_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                return frame
            return crop

    return frame

# ================= PREPROCESS =================
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    return frame

# ================= SEQUENCE =================
def create_sequence(generator):
    while True:
        x_batch, y_batch = next(generator)
        x_seq = []

        for img in x_batch:
            img_uint8 = (img * 255).astype(np.uint8)
            hand_crop = detect_hand(img_uint8)
            hand = preprocess_frame(hand_crop)
            x_seq.append([hand])

        yield np.array(x_seq, dtype=np.float32), y_batch

# ================= ATTENTION =================
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.Wq = self.add_weight(
            shape=(self.dim, self.dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wq'
        )
        self.Wk = self.add_weight(
            shape=(self.dim, self.dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wk'
        )
        self.Wv = self.add_weight(
            shape=(self.dim, self.dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wv'
        )

    def call(self, x):
        Q = tf.matmul(x, self.Wq)
        K = tf.matmul(x, self.Wk)
        V = tf.matmul(x, self.Wv)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.dim, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attn, V)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ================= VGG16 =================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,160,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# ================= MODEL =================
input_layer = Input(shape=(SEQ_LEN, 160, 160, 3))

cnn = TimeDistributed(feature_extractor)(input_layer)
cnn = TimeDistributed(BatchNormalization())(cnn)

attention = SelfAttention()(cnn)
lstm = LSTM(128)(attention)

x = Dense(256, activation='relu')(lstm)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3)
]

# ================= TRAIN =================
train_seq = create_sequence(train_data)
val_seq = create_sequence(val_data)

import math
steps_per_epoch = math.ceil(train_data.samples / BATCH_SIZE)
val_steps = math.ceil(val_data.samples / BATCH_SIZE)


# Phase 1

print("Phase-1 Training Begins ✅")

# Phase 1 → Adam ---->(fast learning)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_seq,
    validation_data=val_seq,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=18,
    callbacks=callbacks,
    workers=1,
    use_multiprocessing=False
)

# Phase 2 (fine-tuning)

print("Phase-2 Fine-tuning Begins ✅")

# Unfreeze last layers
for layer in base_model.layers[-8:]:
    layer.trainable = True

# Phase 2 → SGD ---> (better generalization)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_seq,
    validation_data=val_seq,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=12,
    callbacks=callbacks,
    workers=1,
    use_multiprocessing=False
)

print("Model Training Completed Successfully ✅")

# ================= SAVE MODEL =================
import time
import os

model_name = f"MD_model_{int(time.time())}.h5"   # Saving as .h5 due to compatibility


if os.path.exists(model_name):
    os.remove(model_name)

model.save(model_name, save_format='h5')
print(f"✅ Model saved: {model_name}")

# Labels save
labels = {v: k for k, v in train_data.class_indices.items()}
with open("labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Training Done - Ready for Real-time")