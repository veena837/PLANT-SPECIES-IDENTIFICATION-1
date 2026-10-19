"""
Train the CNN model for Plant Species Identification.

Dataset layout (Kaggle "Indian Medicinal Leaves" or your own images):
    dataset/
        aloevera/   img1.jpg img2.jpg ...
        mint/       ...
        neem-leaf/  ...
        (one folder per plant species)

Run:
    python train_model.py              -> custom CNN (as in the report)
    python train_model.py --transfer   -> MobileNetV2 transfer learning (higher accuracy)
"""
import argparse, json, os, shutil, random
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH = 32
DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("model", "plant_model.keras")
CLASSES_PATH = os.path.join("model", "class_names.json")
REF_DIR = os.path.join("static", "reference")


def build_cnn(num_classes):
    return models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])


def build_transfer(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    return models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def save_reference_images(class_names):
    """Copy one sample image per class to static/reference/ (shown on result page)."""
    os.makedirs(REF_DIR, exist_ok=True)
    for c in class_names:
        folder = os.path.join(DATASET_DIR, c)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            src = os.path.join(folder, random.choice(files))
            shutil.copy(src, os.path.join(REF_DIR, c + ".jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--transfer", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        raise SystemExit("Put your class folders inside the 'dataset' folder first (see README).")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = (build_transfer if args.transfer else build_cnn)(len(class_names))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)

    loss, acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {acc * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join("model", "model_info.json"), "w") as f:
        json.dump({"type": "MobileNetV2" if args.transfer else "Custom CNN",
                   "validation_accuracy": round(acc * 100, 2),
                   "classes": len(class_names)}, f, indent=2)
    save_reference_images(class_names)
    print("Saved model, class names and reference images. Now run: python app.py")


if __name__ == "__main__":
    main()
