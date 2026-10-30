
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "fruit_dataset"
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    class_names=["Fresh", "Rotten"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val",
    class_names=["Fresh", "Rotten"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    class_names=["Fresh", "Rotten"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = tf.keras.layers.Rescaling(
    scale=1 / 127.5,
    offset=-1
)(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model = tf.keras.models.load_model("best_model.keras")

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
]).astype(int)

probabilities = model.predict(test_ds, verbose=0).ravel()

y_pred = (probabilities >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )),
    "recall": float(recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    ))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nTEST PERFORMANCE")

for name, value in metrics.items():
    print(f"{name}: {value * 100:.2f}%")

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Fresh", "Rotten"],
    zero_division=0
))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fresh", "Rotten"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Fruit Freshness Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.legend()
plt.savefig("accuracy_graph.png", dpi=200)
plt.close()

model.save("fruit_freshness_model.keras")

print("\nTraining completed successfully.")
