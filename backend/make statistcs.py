import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import os

# 📁 Dossier de sortie
PREDICTION_DIR = r"C:\Users\ibrahim\Documents\NEW SITE\backend\prediction"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ✅ Chargement du modèle
model = load_model(r"C:\Users\ibrahim\Documents\NEW SITE\backend\model_weights.h5")

# 📚 Classes du dataset
class_names = ['bend', 'fall', 'liedown', 'run', 'sitdown', 'standup', 'walk']

def predict_csi_har(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    probs = model.predict(img_array)[0]
    results = list(zip(class_names, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 📥 Image à prédire
img_path = r"C:\Users\ibrahim\Documents\NEW SITE\backend\output_images\user_1_sample_3_fall_A.png"
predictions = predict_csi_har(img_path)

# 📤 Affichage console
print("Prédictions (classe: probabilité):")
for classe, prob in predictions:
    print(f"{classe}: {prob:.4f}")

# 📄 Sauvegarde en CSV
csv_output_path = os.path.join(PREDICTION_DIR, "resultat_prediction.csv")
df = pd.DataFrame(predictions, columns=['Classe', 'Probabilité'])
df.to_csv(csv_output_path, index=False)

print(f"\n✅ Résultats sauvegardés dans : {csv_output_path}")
