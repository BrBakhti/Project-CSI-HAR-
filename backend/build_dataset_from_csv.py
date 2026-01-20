# backend/build_dataset_from_csv_folders.py
import os
import glob
from tqdm import tqdm
from generate_graph import csv_to_image  # ta fonction existante qui produit un PNG

RAW_DIR = r"C:\Users\ibrahim\Downloads\data CSI"   # <- dossier avec sous-dossiers: bend, fall, ...
OUT_DIR = "dataset_images"                         # <- là où on va écrire les PNG par classe

def main():
    # Crée le dossier de sortie
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    classes.sort()
    if not classes:
        print(f"Aucun sous-dossier de classe trouvé dans: {RAW_DIR}")
        return

    total = 0
    for cls in classes:
        in_dir = os.path.join(RAW_DIR, cls)
        out_dir = os.path.join(OUT_DIR, cls.lower().replace(" ", ""))  
        os.makedirs(out_dir, exist_ok=True)

        csvs = glob.glob(os.path.join(in_dir, "*.csv"))
        if not csvs:
            print(f"[INFO] Aucun CSV pour la classe '{cls}'")
            continue

        for csv_path in tqdm(csvs, desc=f"{cls}"):
            # csv_to_image doit sauvegarder une image dans out_dir et renvoyer son chemin
            # adapte generate_graph.csv_to_image si besoin: csv_to_image(csv_path, out_dir)
            _ = csv_to_image(csv_path, out_dir)
            total += 1

    print(f"\n✔️ Conversion terminée. Total images créées: {total}")
    print(f"Dossier images: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
