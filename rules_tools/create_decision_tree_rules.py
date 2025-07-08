import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Charger le CSV
csv_file = 'rules_tools/features_export.csv'
df = pd.read_csv(csv_file)

# 2. Définir les colonnes de features à utiliser
feature_cols = [
    'bin_pixel_ratio',
    'sacs_autour_ratio',
    'bin_surrounding_diversity',
    'file_size',
    'width',
    'height',
    'avg_red',
    'avg_green',
    'avg_blue',
    'brightness',
    'contrast_level',
    'edge_density',
    'color_diversity',
    'saturation',
    'hue_dominance'
    # Tu peux aussi ajouter les features avancées si elles sont bien extraites dans le CSV
]
X = df[feature_cols]
y = df['label']

# 3. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraîner l'arbre de décision
clf_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_tree.fit(X_train, y_train)

# 5. Afficher les règles sous forme textuelle
rules_text = export_text(clf_tree, feature_names=feature_cols)
print("=== RÈGLES EXTRAITES PAR L'ARBRE DE DÉCISION ===")
print(rules_text)

# 6. Évaluer l'arbre de décision
print("\n=== ÉVALUATION ARBRE DE DÉCISION ===")
y_pred_tree = clf_tree.predict(X_test)
print(classification_report(y_test, y_pred_tree))

# 7. Sauvegarder les règles dans un fichier
with open("rules_tools/decision_tree_rules.txt", "w") as f:
    f.write(rules_text)