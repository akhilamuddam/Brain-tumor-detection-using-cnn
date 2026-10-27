
import os
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

train_dir = os.path.join(DATASET_DIR, "Training")
test_dir = os.path.join(DATASET_DIR, "Testing")

if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        "Create dataset/Training and dataset/Testing folders "
        "with tumor and no_tumor subfolders."
    )

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["no_tumor", "tumor"],
    seed=42
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["no_tumor", "tumor"],
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.keras")

print("Model saved to models/brain_tumor_model.keras")
print("Class mapping:", train_data.class_names)
