# Plant Species Identification (Flask + CNN)

Image-based plant species identification with a CNN (TensorFlow/Keras), Flask backend,
User module (upload -> predict -> name / region / medicinal uses / confidence) and
Admin dashboard (stats, pie chart, history, Correct/Wrong verification).

## 1. Setup (VS Code terminal, Windows)
```
python --version            # use Python 3.10 or 3.11 (TensorFlow 2.16 does not support 3.13)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add the dataset
Download "Indian Medicinal Leaves" from Kaggle (link in your report, reference [1]).
Put ONE FOLDER PER SPECIES inside `dataset/`:
```
dataset/aloevera/*.jpg
dataset/mint/*.jpg
dataset/neem-leaf/*.jpg   ... etc
```
Tip: start with 8-10 species (aloevera, arali, curryleaf, mint, neem-leaf, pumpkin, rose, tamarind) - training is faster.

## 3. Train the model (one time)
```
python train_model.py              # custom CNN, as in the report
python train_model.py --transfer   # MobileNetV2, usually much higher accuracy
```
This creates `model/plant_model.keras`, `model/class_names.json`, `model/model_info.json`
and copies one reference image per species into `static/reference/`.

## 4. Run
```
python app.py
```
Open http://127.0.0.1:5000  (user)  and  http://127.0.0.1:5000/admin  (admin, no login in the prototype).

## Adding plant details
Edit `plant_data.json`. The key is the folder name in lowercase with `-`/`_` removed
(`neem-leaf` -> `neemleaf`). Species without an entry still work, but show "Details not added yet".
Medicinal information is educational only - verify it from a reliable source before presenting.

## Project structure
```
app.py               Flask routes (user + admin)
train_model.py       CNN training
plant_data.json      names, scientific names, regions, medicinal uses
prediction_history.json   created automatically
templates/ static/   HTML, CSS, JS, uploads, reference images
schema.sql           optional MySQL tables from the report
```
Map and chart need internet (Leaflet / Chart.js / Bootstrap from CDN).
