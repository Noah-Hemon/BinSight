import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import SimpleClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Charger le CSV
csv_file = './rules_tools/features_export.csv'
df = pd.read_csv(csv_file)

# Colonnes de features à utiliser
feature_cols = [
    'bin_pixel_ratio', 'sacs_autour_ratio', 'bin_surrounding_diversity',
    'file_size', 'width', 'height', 'avg_red', 'avg_green', 'avg_blue',
    'brightness', 'contrast_level', 'edge_density', 'color_diversity',
    'saturation', 'hue_dominance'
]

clf = SimpleClassifier()
clf.set_mode('rules')

y_true = []
y_pred = []

for idx, row in df.iterrows():
    features = {col: row[col] for col in feature_cols}
    real_label = row['label']
    pred, conf = clf.classify_rules(features)[:2]
    y_true.append(real_label)
    y_pred.append(pred)

# Affichage des métriques
print("=== METRIQUES CLASSIFY_RULES (arbre de décision hardcodé) ===")
print(classification_report(y_true, y_pred, digits=3))
print("\nMatrice de confusion :")
print(confusion_matrix(y_true, y_pred)) 