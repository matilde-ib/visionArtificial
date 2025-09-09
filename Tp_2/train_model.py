# train_model.py (solo Hu)
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "machine" / "generated-files"
CSV_PATH   = DATA_DIR / "shapes-hu-moments.csv"
MODEL_PATH = SCRIPT_DIR / "modelo.joblib"

print("[CSV]", CSV_PATH)
print("[MODEL OUT]", MODEL_PATH)

# Chequeo
if not CSV_PATH.exists():
    sys.exit(f"[ERROR] No se encuentra el CSV: {CSV_PATH}")

# Cargar datos
df = pd.read_csv(CSV_PATH)

# Solo Hu (color-agnóstico)
hu_cols = [f"hu{i}" for i in range(1, 8)]
if not all(c in df.columns for c in hu_cols):
    sys.exit("[ERROR] Faltan columnas hu1..hu7 en el CSV.")

feature_cols = hu_cols
X = df[feature_cols].values
y = df["label"].values  # 'corazon'/'circulo'/'pentagono'

print("[INFO] Features usadas:", feature_cols)
print("[INFO] Balance de clases:\n", df["label"].value_counts())

# Split, entrenamiento y evaluación
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
clf = DecisionTreeClassifier(max_depth=10, random_state=0)
clf.fit(Xtr, ytr)

print("\nReporte de test:\n", classification_report(yte, clf.predict(Xte)))
print("Matriz de confusión:\n", confusion_matrix(yte, clf.predict(Xte)))

# Guardar modelo
dump(clf, MODEL_PATH)
print(f"[OK] Guardado modelo en: {MODEL_PATH}")
