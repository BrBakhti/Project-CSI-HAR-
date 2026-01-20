# predict_csi_har.py
import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

HERE = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(HERE, "model_weights.h5")
CLASSES_PATH = os.path.join(HERE, "classes.json")

# Mapping par défaut si classes.json absent
DEFAULT_CLASSES = ["bend", "fall", "run", "sitdown", "standup", "walk", "liedown"]

# Charger les labels depuis classes.json si présent
if os.path.exists(CLASSES_PATH):
    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
        if not isinstance(CLASS_NAMES, list) or not CLASS_NAMES:
            CLASS_NAMES = DEFAULT_CLASSES
    except Exception:
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# Import de preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_pp
try:
    from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_pp
except:
    mobilenet_pp = mobilenet_v2_pp
try:
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_pp
except:
    mobilenet_v3_pp = mobilenet_v2_pp
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_pp
from tensorflow.keras.applications.xception import preprocess_input as xception_pp
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_pp
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pp
except:
    efficientnet_pp = mobilenet_v2_pp

_PREPROCESS_CANDIDATES = [
    ("mobilenet_v2", mobilenet_v2_pp),
    ("mobilenet", mobilenet_pp),
    ("mobilenet_v3", mobilenet_v3_pp),
    ("resnet50", resnet50_pp),
    ("xception", xception_pp),
    ("inception_v3", inception_v3_pp),
    ("efficientnet", efficientnet_pp),
    ("identity", lambda x: x),
]

_model = None
_input_size = (160, 160)
_used_pp_name = "unknown"

def _try_load_model():
    last_err = None
    for name, fn in _PREPROCESS_CANDIDATES:
        try:
            m = load_model(MODEL_PATH, custom_objects={"preprocess_input": fn}, compile=False)
            print(f"[predict] model loaded with custom preprocess_input = {name}")
            return m, name
        except Exception as e:
            last_err = e
    raise last_err

def _ensure_model():
    global _model, _input_size, _used_pp_name
    if _model is None:
        _model, _used_pp_name = _try_load_model()
        try:
            ishape = _model.input_shape
            if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
                ishape = ishape[0]
            if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
                _input_size = (int(ishape[1]), int(ishape[2]))
            print(f"[predict] inferred input size = {_input_size}")
        except Exception:
            print("[predict] could not infer input size; using default", _input_size)

def predict_image(image_path: str):
    """Retourne (label, confiance_en_%)."""
    _ensure_model()

    img = load_img(image_path, target_size=_input_size)
    x = img_to_array(img)
    x = np.expand_dims(x, 0)
    x = x / 255.0

    probs = _model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx]) * 100.0

    label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
    return label, conf
