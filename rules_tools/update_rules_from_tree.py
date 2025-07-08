import sqlite3
import json

# Traduction des features pour la description
feature_fr = {
    'height': 'hauteur',
    'color_diversity': 'diversité couleur',
    'avg_green': 'vert moyen',
    'avg_blue': 'bleu moyen',
    'saturation': 'saturation',
    'contrast_level': 'contraste',
    'edge_density': 'densité de contours',
    'file_size': 'taille fichier',
    'bin_surrounding_diversity': 'diversité autour poubelle',
}

# Traduction des opérateurs
op_fr = {
    '>': '>',
    '<': '<',
    '>=': '≥',
    '<=': '≤',
    '==': '=',
    '!=': '≠',
}

# Règles issues de l'arbre de décision (exemple)
rules = [
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": "<=", "value": 0.80},
            "avg_green": {"operator": "<=", "value": 162.38},
            "avg_blue": {"operator": "<=", "value": 146.84}
        },
        "action": "pleine",
        "priority": 1
    },
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": "<=", "value": 0.80},
            "avg_green": {"operator": "<=", "value": 162.38},
            "avg_blue": {"operator": ">", "value": 146.84}
        },
        "action": "pleine",
        "priority": 2
    },
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": "<=", "value": 0.80},
            "avg_green": {"operator": ">", "value": 162.38}
        },
        "action": "vide",
        "priority": 3
    },
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": ">", "value": 0.80},
            "saturation": {"operator": "<=", "value": 37.06}
        },
        "action": "vide",
        "priority": 4
    },
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": ">", "value": 0.80},
            "saturation": {"operator": ">", "value": 37.06},
            "avg_blue": {"operator": "<=", "value": 107.11}
        },
        "action": "vide",
        "priority": 5
    },
    {
        "condition_json": {
            "height": {"operator": "<=", "value": 425.5},
            "color_diversity": {"operator": ">", "value": 0.80},
            "saturation": {"operator": ">", "value": 37.06},
            "avg_blue": {"operator": ">", "value": 107.11}
        },
        "action": "pleine",
        "priority": 6
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": "<=", "value": 76.50},
            "avg_green": {"operator": "<=", "value": 50.33}
        },
        "action": "vide",
        "priority": 7
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": "<=", "value": 76.50},
            "avg_green": {"operator": ">", "value": 50.33},
            "contrast_level": {"operator": "<=", "value": 65.92}
        },
        "action": "pleine",
        "priority": 8
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": "<=", "value": 76.50},
            "avg_green": {"operator": ">", "value": 50.33},
            "contrast_level": {"operator": ">", "value": 65.92}
        },
        "action": "vide",
        "priority": 9
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": ">", "value": 76.50},
            "edge_density": {"operator": "<=", "value": 0.20},
            "file_size": {"operator": "<=", "value": 652966.00}
        },
        "action": "vide",
        "priority": 10
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": ">", "value": 76.50},
            "edge_density": {"operator": "<=", "value": 0.20},
            "file_size": {"operator": ">", "value": 652966.00}
        },
        "action": "pleine",
        "priority": 11
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": ">", "value": 76.50},
            "edge_density": {"operator": ">", "value": 0.20},
            "bin_surrounding_diversity": {"operator": "<=", "value": 0.00}
        },
        "action": "vide",
        "priority": 12
    },
    {
        "condition_json": {
            "height": {"operator": ">", "value": 425.5},
            "avg_blue": {"operator": ">", "value": 76.50},
            "edge_density": {"operator": ">", "value": 0.20},
            "bin_surrounding_diversity": {"operator": ">", "value": 0.00}
        },
        "action": "pleine",
        "priority": 13
    }
]

# Générer une description lisible pour chaque règle
for idx, rule in enumerate(rules, 1):
    conds = []
    for feat, cond in rule["condition_json"].items():
        feat_txt = feature_fr.get(feat, feat)
        op_txt = op_fr.get(cond["operator"], cond["operator"])
        val_txt = cond["value"]
        conds.append(f"{feat_txt} {op_txt} {val_txt}")
    rule["description"] = " et ".join(conds)
    rule["name"] = f"Règle {idx} (auto)"

# Connexion à la BDD
conn = sqlite3.connect('binsight.db')
cursor = conn.cursor()

# Supprimer les anciennes règles
cursor.execute("DELETE FROM classification_rules")

# Insérer les nouvelles règles
for rule in rules:
    cursor.execute('''
        INSERT INTO classification_rules (name, description, condition_json, action, priority)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        rule["name"],
        rule["description"],
        json.dumps(rule["condition_json"]),
        rule["action"],
        rule["priority"]
    ))

conn.commit()
conn.close()
print(f"✅ {len(rules)} règles insérées dans la BDD (features simples, pas chemins d'arbre)") 