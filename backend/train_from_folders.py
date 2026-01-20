import os, json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Chemin du dataset ===
DATA_DIR = "dataset_images"  # <-- ton dossier avec sous-dossiers par classe

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS_HEAD = 12         # entraînement tête
EPOCHS_FT = 6            # fine-tuning
MODEL_OUT = "model_best.keras"  # ✅ format moderne
CLASSES_JSON = "classes.json"
SEED = 42

# --- Chargement des datasets ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

class_names = train_ds.class_names
print("Classes :", class_names)

# Sauvegarde de la liste des classes
with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)
print("Sauvegardé ->", CLASSES_JSON)

# Prefetch pour accélérer
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# --- Modèle ---
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.layers.Lambda(preprocess_input)(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- Callbacks ---
ckpt = tf.keras.callbacks.ModelCheckpoint(
    MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
)
es   = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

# --- Phase 1 : entraînement de la tête ---
print("\n=== Phase 1 : Entraînement de la tête ===")
history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=[ckpt, es])

# --- Phase 2 : fine-tuning ---
print("\n=== Phase 2 : Fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]:  # on ne dégèle que les dernières couches
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=[ckpt, es])

# --- Fusion des historiques ---
def merge_histories(h1, h2):
    merged = {}
    for k in h1.history.keys():
        merged[k] = h1.history[k] + h2.history[k]
    return merged

history = merge_histories(history_head, history_ft)

# --- Tracés des courbes ---
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="train_acc")
plt.plot(history["val_accuracy"], label="val_acc")
plt.title("Évolution de l'accuracy")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history["loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Évolution de la perte (loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# --- Évaluation finale ---
model.load_weights(MODEL_OUT)

y_true, y_pred = [], []
for xb, yb in val_ds:
    logits = model.predict(xb, verbose=0)
    y_true.extend(np.argmax(yb.numpy(), axis=1))
    y_pred.extend(np.argmax(logits, axis=1))

print("\n=== Rapport de classification ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

print("\n=== Matrice de confusion (texte) ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# --- Affichage graphique de la matrice de confusion ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matrice de confusion - Validation set")
plt.tight_layout()
plt.show()

# (Optionnel) Sauvegarde
# plt.savefig("confusion_matrix.png", dpi=300)

print("\n✅ Modèle sauvegardé ->", MODEL_OUT)
