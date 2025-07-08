import os
import csv
from app import feature_extractor, classifier
import sqlite3

# Dossiers à parcourir
DATA_DIR = 'Data/train'
LABELLED_DIRS = [
    ('with_label/dirty', 'pleine'),
    ('with_label/clean', 'vide'),
]
# On ignore le dossier no_label

# Extensions d'image acceptées
IMG_EXTS = {'.jpg', '.jpeg', '.png'}

rows = []

# Images labellisées uniquement
for subdir, label in LABELLED_DIRS:
    dir_path = os.path.join(DATA_DIR, subdir)
    if not os.path.exists(dir_path):
        continue
    for fname in os.listdir(dir_path):
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            features = feature_extractor.extract(fpath)
            if features is None:
                continue
            result = classifier.classify_rules(features, label)
            if len(result) == 3:
                auto_class, conf, is_correct = result
            else:
                auto_class, conf = result
                is_correct = ''
            rows.append([
                fname,
                label,
                features.get('bin_pixel_ratio', ''),
                features.get('sacs_autour_ratio', ''),
                features.get('bin_surrounding_diversity', ''),
                features.get('file_size', ''),
                features.get('width', ''),
                features.get('height', ''),
                features.get('avg_red', ''),
                features.get('avg_green', ''),
                features.get('avg_blue', ''),
                features.get('brightness', ''),
                features.get('contrast_level', ''),
                features.get('edge_density', ''),
                features.get('color_diversity', ''),
                features.get('saturation', ''),
                features.get('hue_dominance', ''),
                auto_class,
                conf,
                is_correct
            ])
        except Exception as e:
            print(f"Erreur sur {fpath} : {e}")

with open('features_export.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'filename', 'label', 'bin_pixel_ratio', 'sacs_autour_ratio', 'bin_surrounding_diversity',
        'file_size', 'width', 'height', 'avg_red', 'avg_green', 'avg_blue', 'brightness',
        'contrast_level', 'edge_density', 'color_diversity', 'saturation', 'hue_dominance',
        'auto_classification', 'confidence', 'is_correct'
    ])
    writer.writerows(rows)

print(f"Exporté {len(rows)} images labellisées dans features_export.csv")

if __name__ == "__main__":
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    cursor.execute("SELECT auto_classification, COUNT(*) FROM images WHERE comment = 'Importation en masse' GROUP BY auto_classification;")
    results = cursor.fetchall()
    print("Statut des images importées en masse :")
    for label, count in results:
        print(f"{label}: {count}")
    conn.close() 