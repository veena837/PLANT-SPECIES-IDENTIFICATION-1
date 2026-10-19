"""
Plant Species Identification - Flask application
User module : upload leaf image -> CNN prediction -> details / region / medicinal uses
Admin module: dashboard, prediction history, verify Correct/Wrong, accuracy chart
"""
import json, os, re, uuid
from datetime import datetime

import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify)
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
REF_DIR = os.path.join(BASE, "static", "reference")
MODEL_PATH = os.path.join(BASE, "model", "plant_model.keras")
CLASSES_PATH = os.path.join(BASE, "model", "class_names.json")
PLANT_DATA_PATH = os.path.join(BASE, "plant_data.json")
HISTORY_PATH = os.path.join(BASE, "prediction_history.json")
IMG_SIZE = (224, 224)
ALLOWED = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.secret_key = "plant-species-secret"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# approximate map coordinates for state pins
STATE_COORDS = {
    "Karnataka": (14.5, 75.7), "Tamil Nadu": (11.1, 78.6), "Maharashtra": (19.7, 75.7),
    "Uttar Pradesh": (26.8, 80.9), "Rajasthan": (27.0, 74.2), "Gujarat": (22.3, 71.2),
    "Kerala": (10.8, 76.3), "Andhra Pradesh": (15.9, 79.7), "West Bengal": (22.9, 87.8),
    "Punjab": (31.1, 75.3), "Uttarakhand": (30.1, 79.0), "Bihar": (25.1, 85.3),
    "Jammu & Kashmir": (33.8, 76.6),
}

_model = None
_classes = []
_plants = {}


def norm(s):
    """'neem-leaf' / 'Neem_Leaf' / 'aloe_vera' -> 'neemleaf' / 'aloevera'"""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_resources():
    global _model, _classes, _plants
    with open(PLANT_DATA_PATH, encoding="utf-8") as f:
        _plants = json.load(f)
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        import tensorflow as tf  # imported here so the site still opens without a model
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
            safe_mode=False,
        )
        with open(CLASSES_PATH) as f:
            _classes = json.load(f)
        print("Model loaded with classes:", _classes)
    else:
        print("WARNING: model not found. Run  python train_model.py  first.")


def plant_info(class_name):
    key = norm(class_name)
    if key in _plants:
        return _plants[key]
    # fallback for classes without an entry in plant_data.json
    return {"name": class_name.replace("_", " ").replace("-", " ").title(),
            "scientific_name": "Not available", "description": "Details not added yet.",
            "regions": [], "medicinal_uses": []}


def reference_url(class_name):
    p = os.path.join(REF_DIR, class_name + ".jpg")
    return url_for("static", filename=f"reference/{class_name}.jpg") if os.path.exists(p) else None


def read_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def write_history(h):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2)


def preprocess(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)   # resize + RGB
    arr = np.array(img, dtype="float32")                      # numerical array
    return np.expand_dims(arr, axis=0)                        # add batch dimension
    # (normalisation / rescaling happens inside the model's first layer)


def confidence_label(c):
    if c >= 90: return "Highly Confident"
    if c >= 70: return "Moderately Confident"
    return "Low Confidence Prediction"


# ------------------------------------------------------------------ USER MODULE
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose a plant leaf image first.")
        return redirect(url_for("index"))
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED:
        flash("Unsupported file. Please upload a JPG, JPEG or PNG image.")
        return redirect(url_for("index"))
    if _model is None:
        flash("Model not trained yet. Run 'python train_model.py' and restart the app.")
        return redirect(url_for("index"))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    file.save(path)
    try:
        x = preprocess(path)
    except Exception:
        os.remove(path)
        flash("Could not read the image. It may be corrupted.")
        return redirect(url_for("index"))

    probs = _model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    class_name = _classes[idx]
    confidence = round(float(probs[idx]) * 100, 2)

    history = read_history()
    history.append({
        "id": uuid.uuid4().hex[:8],
        "image": fname,
        "predicted": class_name,
        "actual": None,
        "confidence": confidence,
        "status": "Pending",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_history(history)
    return redirect(url_for("result", cls=class_name, img=fname, conf=confidence))


@app.route("/result")
def result():
    cls = request.args.get("cls", "")
    img = request.args.get("img", "")
    conf = float(request.args.get("conf", 0))
    info = plant_info(cls)
    return render_template("result.html", cls=cls, info=info, img=img, conf=conf,
                           label=confidence_label(conf), ref=reference_url(cls))


@app.route("/region/<cls>")
def region(cls):
    info = plant_info(cls)
    pins = [{"state": s, "lat": STATE_COORDS[s][0], "lng": STATE_COORDS[s][1]}
            for s in info["regions"] if s in STATE_COORDS]
    return render_template("region.html", cls=cls, info=info, pins=pins)


@app.route("/medicinal/<cls>")
def medicinal(cls):
    return render_template("medicinal.html", cls=cls, info=plant_info(cls))


# ----------------------------------------------------------------- ADMIN MODULE
@app.route("/admin")
def admin():
    history = list(reversed(read_history()))          # latest first
    correct = sum(1 for h in history if h["status"] == "Correct")
    wrong = sum(1 for h in history if h["status"] == "Wrong")
    pending = len(history) - correct - wrong
    evaluated = correct + wrong
    accuracy = round(correct / evaluated * 100, 1) if evaluated else 0
    for h in history:
        h["level"] = confidence_label(h["confidence"])
        h["predicted_name"] = plant_info(h["predicted"])["name"]
        h["actual_name"] = plant_info(h["actual"])["name"] if h["actual"] else "Not Set"
    info_path = os.path.join(BASE, "model", "model_info.json")
    model_info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return render_template("admin.html", history=history, total=len(history),
                           correct=correct, wrong=wrong, pending=pending,
                           accuracy=accuracy, evaluated=evaluated,
                           model_ok=_model is not None, model_info=model_info,
                           retrain=(evaluated >= 5 and accuracy < 80))


@app.route("/admin/verify", methods=["POST"])
def verify():
    pid = request.form.get("id")
    action = request.form.get("action")
    history = read_history()
    for h in history:
        if h["id"] == pid:
            if action == "correct":
                h["status"], h["actual"] = "Correct", h["predicted"]
            elif action == "wrong":
                h["status"] = "Wrong"
                actual = request.form.get("actual")
                if actual:
                    h["actual"] = actual
    write_history(history)
    return redirect(url_for("admin"))


@app.route("/admin/delete/<pid>", methods=["POST"])
def delete_record(pid):
    write_history([h for h in read_history() if h["id"] != pid])
    return redirect(url_for("admin"))


@app.route("/api/classes")
def api_classes():
    return jsonify(_classes)


@app.context_processor
def inject_helpers():
    return {"all_classes": _classes}


load_resources()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)