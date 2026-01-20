# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, shutil, json, time

# --- import robuste pour csv_to_image ---
try:
    from generate_graph import csv_to_image
except Exception:
    from generate_graph import csv_to_image

from predict_csi_har import predict_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'csi')
OUTPUT_FOLDER = os.path.join(UPLOAD_FOLDER, 'output_images')
RESULTS_JSON  = os.path.join(BASE_DIR, 'csi_results.json')
CLASSES_PATH  = os.path.join(BASE_DIR, 'classes.json')

CLASS_NAMES = None
if os.path.exists(CLASSES_PATH):
    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = None

def normalize_activity(activity):
    try:
        idx = int(activity)
        if CLASS_NAMES and 0 <= idx < len(CLASS_NAMES):
            return CLASS_NAMES[idx]
        return str(activity)
    except Exception:
        return activity if activity not in (None, "") else "-"

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Assurer un JSON valide dès le départ
if not os.path.exists(RESULTS_JSON) or os.path.getsize(RESULTS_JSON) == 0:
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump([], f)

def _load_results_safe():
    if not os.path.exists(RESULTS_JSON):
        return []
    try:
        with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        try:
            shutil.copyfile(RESULTS_JSON, RESULTS_JSON + ".bak")
        except Exception:
            pass
        print("[WARN] csi_results.json illisible:", e)
        return []

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def _versioned_output_url(path_like: str) -> str:
    if not path_like:
        return ""
    p = path_like.replace("\\", "/")
    if p.startswith("/output_images/"):
        web = p; rel = p[len("/output_images/"):]
    elif p.startswith("output_images/"):
        web = "/" + p; rel = p[len("output_images/"):]
    elif "output_images/" in p:
        rel = p.split("output_images/", 1)[1]; web = f"/output_images/{rel}"
    else:
        rel = os.path.basename(p); web = f"/output_images/{rel}"
    abs_path = os.path.join(OUTPUT_FOLDER, rel.replace("/", os.sep))
    ver = str(int(os.path.getmtime(abs_path))) if os.path.exists(abs_path) else ""
    return f"{web}?v={ver}" if ver else web

# === 1) Upload + analyse ===
@app.route('/upload-csv-folder', methods=['POST'])
def upload_csv_folder():
    # 1) récupérer les fichiers (supporte "files" ou "file")
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        one = request.files.get("file")
        if one:
            uploaded_files = [one]
    if not uploaded_files:
        return jsonify({"message": "Aucun fichier reçu (clé 'files' ou 'file').", "results": []}), 400

    # 2) vider et recréer les dossiers d’upload/sortie (lot courant uniquement)
    if os.path.exists(UPLOAD_FOLDER): shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if os.path.exists(OUTPUT_FOLDER): shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 3) traiter
    results, errors = [], []
    for file in uploaded_files:
        try:
            csv_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(csv_path)

            img_full_path = csv_to_image(csv_path, OUTPUT_FOLDER)
            if not os.path.exists(img_full_path):
                raise RuntimeError(f"image non générée: {img_full_path}")

            activity, confidence = predict_image(img_full_path)
            img_name  = os.path.basename(img_full_path)
            image_url = _versioned_output_url(f"output_images/{img_name}")

            results.append({
                "csv_file": file.filename,
                "image_path": image_url,            # ex: /output_images/xxx.png?v=...
                "activity": activity,               # label ou index mappé
                "confidence": round(float(confidence), 2)
            })

            # logs utiles
            print(f"[OK] {file.filename} -> {img_name} | {activity} {confidence:.2f}%")

        except Exception as e:
            msg = f"{file.filename}: {e}"
            errors.append(msg)
            print("[ERROR]", msg)

    # 4) écrire le JSON (ATOMIQUE) avec **uniquement** le lot courant
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_JSON)

    # 5) infos de vérification (mtime)
    mtime = os.path.getmtime(RESULTS_JSON)
    mtime_h = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

    status = 207 if errors and results else (400 if errors and not results else 200)
    return jsonify({
        "message": "Analyse terminée" if results else "Aucun résultat valide",
        "results": results,
        "errors": errors,
        "json_path": RESULTS_JSON,
        "json_mtime": mtime,
        "json_mtime_readable": mtime_h
    }), status

# === 2) Stats + résultats ===
@app.route('/csi-stats', methods=['GET'])
def csi_stats():
    raw_results = _load_results_safe()

    results = []
    for it in raw_results:
        results.append({
            "csv_file": it.get("csv_file", ""),
            "image_path": _versioned_output_url(it.get("image_path") or ""),
            "activity": normalize_activity(it.get("activity")),
            "confidence": float(it.get("confidence", 0.0)),
        })

    stats = {}
    for it in results:
        act = it["activity"] or "-"
        conf = float(it["confidence"])
        if act not in stats:
            stats[act] = {"count": 0, "total_confidence": 0.0}
        stats[act]["count"] += 1
        stats[act]["total_confidence"] += conf

    activities, most_frequent, max_count = [], None, 0
    for act, v in stats.items():
        avg = v["total_confidence"] / max(v["count"], 1)
        activities.append({"activity": act, "count": v["count"], "average_confidence": round(avg, 2)})
        if v["count"] > max_count:
            most_frequent, max_count = act, v["count"]

    try:
        mtime = os.path.getmtime(RESULTS_JSON)
        mtime_h = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    except FileNotFoundError:
        mtime, mtime_h = 0, None

    summary = {"total_files": len(results), "activities": activities, "most_frequent": most_frequent}
    return jsonify({"summary": summary, "results": results,
                    "debug": {"json_path": RESULTS_JSON, "json_mtime": mtime, "json_mtime_readable": mtime_h}}), 200

@app.route('/output_images/<path:filename>')
def serve_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    print("✅ Backend CSI prêt sur http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
