# backend/test_upload.py

import requests
import os

# 🖼️ Dossier où tu mets tes .csv de test
csv_folder = "uploads/csi"
url = "http://127.0.0.1:5000/upload-csv-folder"

# ✅ Vérifie que le dossier existe
if not os.path.exists(csv_folder):
    print(f"❌ Dossier introuvable : {csv_folder}")
    exit(1)

# ✅ Récupère les fichiers .csv
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
if not csv_files:
    print("❌ Aucun fichier CSV trouvé dans le dossier.")
    exit(1)

# 📤 Rassembler les fichiers à envoyer avec ouverture sécurisée
files = []
for file in csv_files:
    filepath = os.path.join(csv_folder, file)
    with open(filepath, 'rb') as f:
        files.append(('files', (file, f.read(), 'text/csv')))

# 🚀 Envoi POST
try:
    response = requests.post(url, files=files)
except Exception as e:
    print("❌ Erreur lors de l’envoi :", e)
    exit(1)

# ✅ Affichage du vrai contenu renvoyé
print("✅ Réponse brute du serveur :")
print("Status code :", response.status_code)
print("Texte brut  :", response.text)

# ✅ Tentative de décodage JSON (optionnel)
try:
    print("✅ JSON décodé :", response.json())
except Exception as e:
    print("❌ Erreur JSON :", e)
