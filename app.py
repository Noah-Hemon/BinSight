from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import sqlite3
from werkzeug.utils import secure_filename
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import requests 
import shutil 
import random 
import json
import traceback
import numpy as np
import cv2
from datetime import datetime
from flask_socketio import SocketIO, emit # NOUVELLE IMPORTATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.model_selection import cross_val_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'binsight-secret-key-2023'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# NOUVEAU: Initialisation de SocketIO
socketio = SocketIO(app)

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

def parse_db_timestamp(timestamp_str):
    """
    NOUVEAU: Analyse un timestamp de la BDD, qui peut avoir ou non des microsecondes.
    """
    if not timestamp_str:
        return None
    try:
        # Essayer avec les microsecondes
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # Sinon, essayer sans
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

def migrate_database():
    """Migrer la base de données pour ajouter les colonnes manquantes"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    try:
        # Vérifier la structure actuelle de la table
        cursor.execute("PRAGMA table_info(images)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Colonnes existantes dans la BDD: {columns}")
        
        # Définir toutes les colonnes attendues et leur type
        expected_columns = {
            'user_annotation': 'TEXT',
            'histogram_features': 'TEXT',
            'texture_features': 'TEXT',
            'shape_features': 'TEXT',
            'color_diversity': 'REAL',
            'saturation': 'REAL',
            'hue_dominance': 'REAL'
        }

        # Ajouter les colonnes manquantes une par une
        for col_name, col_type in expected_columns.items():
            if col_name not in columns:
                print(f"Ajout de la colonne manquante: {col_name}...")
                cursor.execute(f'ALTER TABLE images ADD COLUMN {col_name} {col_type}')
                conn.commit()
                print(f"✓ Colonne {col_name} ajoutée avec succès.")
            else:
                print(f"✓ Colonne {col_name} déjà présente.")
            
        conn.close()
        return True
        
    except sqlite3.OperationalError as e:
        print(f"Erreur lors de la migration: {e}")
        print("Recréation complète de la base de données...")
        conn.close()
        
        # Supprimer et recréer la base
        if os.path.exists('binsight.db'):
            os.remove('binsight.db')
        init_db()
        return True
        
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        conn.close()
        return False

def init_db():
    """Initialiser la base de données avec la structure complète"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    # Table principale des images
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            manual_annotation TEXT,
            auto_classification TEXT,
            user_annotation TEXT,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            avg_red REAL,
            avg_green REAL,
            avg_blue REAL,
            contrast_level REAL,
            brightness REAL,
            edge_density REAL,
            location TEXT,
            bin_type TEXT,
            comment TEXT,
            confidence REAL DEFAULT 0.0,
            -- Nouvelles colonnes pour les caractéristiques avancées
            histogram_features TEXT,
            texture_features TEXT,
            shape_features TEXT,
            color_diversity REAL,
            saturation REAL,
            hue_dominance REAL
        )
    ''')
    
    # Table pour les règles de classification configurables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            condition_json TEXT NOT NULL,
            action TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            active BOOLEAN DEFAULT 1,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            modified_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table pour les métadonnées des annotations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotation_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            annotation_type TEXT,
            metadata_json TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Base de données initialisée avec la structure complète")
    
    # Ajouter des règles par défaut
    add_default_rules()

def add_default_rules():
    """Ajouter des règles de classification par défaut"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    # Vérifier si des règles existent déjà
    cursor.execute("SELECT COUNT(*) FROM classification_rules")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    default_rules = [
        {
            'name': 'Poubelle pleine - Faible luminosité',
            'description': 'Détecte les poubelles pleines basé sur la faible luminosité',
            'condition_json': '{"brightness": {"operator": "<", "value": 80}, "edge_density": {"operator": ">", "value": 0.15}}',
            'action': 'full',
            'priority': 1
        },
        {
            'name': 'Poubelle vide - Haute luminosité',
            'description': 'Détecte les poubelles vides basé sur la haute luminosité',
            'condition_json': '{"brightness": {"operator": ">", "value": 150}, "contrast_level": {"operator": "<", "value": 40}}',
            'action': 'empty',
            'priority': 2
        },
        {
            'name': 'Poubelle pleine - Texture complexe',
            'description': 'Détecte les poubelles pleines basé sur la complexité de texture',
            'condition_json': '{"texture_features.entropy": {"operator": ">", "value": 0.5}, "color_diversity": {"operator": ">", "value": 0.3}}',
            'action': 'full',
            'priority': 3
        }
    ]
    
    for rule in default_rules:
        cursor.execute('''
            INSERT INTO classification_rules (name, description, condition_json, action, priority)
            VALUES (?, ?, ?, ?, ?)
        ''', (rule['name'], rule['description'], rule['condition_json'], rule['action'], rule['priority']))
    
    conn.commit()
    conn.close()
    print("✓ Règles de classification par défaut ajoutées")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class FeatureExtractor:
    """Extraire les caractéristiques d'une image."""
    def extract(self, image_path):
        try:
            print(f"Extraction des caractéristiques pour: {image_path}")
            
            # Vérifier que le fichier existe
            if not os.path.exists(image_path):
                raise Exception(f"Fichier non trouvé: {image_path}")
            
            # Taille du fichier
            file_size = os.path.getsize(image_path)
            print(f"Taille du fichier: {file_size} bytes")
            
            # Ouvrir avec PIL et OpenCV
            pil_image = Image.open(image_path)
            width, height = pil_image.size
            print(f"Dimensions: {width}x{height}")
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Couleurs moyennes
            np_image = np.array(pil_image)
            avg_red = np.mean(np_image[:, :, 0])
            avg_green = np.mean(np_image[:, :, 1])
            avg_blue = np.mean(np_image[:, :, 2])
            
            # Luminosité
            brightness = np.mean(np_image)
            
            # Nouvelles caractéristiques avancées
            advanced_features = self.extract_advanced_features(image_path, np_image)
            
            features = {
                'file_size': file_size,
                'width': width,
                'height': height,
                'avg_red': float(avg_red),
                'avg_green': float(avg_green),
                'avg_blue': float(avg_blue),
                'brightness': float(brightness),
                'contrast_level': float(advanced_features['contrast']),
                'edge_density': float(advanced_features['edge_density']),
                'histogram_features': advanced_features['histogram'],
                'texture_features': advanced_features['texture'],
                'shape_features': advanced_features['shape'],
                'color_diversity': float(advanced_features['color_diversity']),
                'saturation': float(advanced_features['saturation']),
                'hue_dominance': float(advanced_features['hue_dominance'])
            }
            
            print(f"✓ Caractéristiques extraites: {len(features)} features")
            return features
            
        except Exception as e:
            print(f"✗ Erreur lors de l'extraction: {e}")
            traceback.print_exc()
            return None
    
    def extract_advanced_features(self, image_path, np_image):
        """Extraire des caractéristiques avancées - VERSION AMÉLIORÉE"""
        try:
            # Charger l'image avec OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise Exception("Impossible de charger l'image avec OpenCV")
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 1. Histogrammes de couleur améliorés
            hist_r = cv2.calcHist([cv_image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([cv_image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([cv_image], [2], None, [256], [0, 256])
            
            # Caractéristiques d'histogramme plus sophistiquées
            hist_features = {
                'red_mean': float(np.mean(hist_r.astype(np.float32).ravel())),
                'green_mean': float(np.mean(hist_g.astype(np.float32).ravel())),
                'blue_mean': float(np.mean(hist_b.astype(np.float32).ravel())),
                'red_std': float(np.std(hist_r.astype(np.float32).ravel())),
                'green_std': float(np.std(hist_g.astype(np.float32).ravel())),
                'blue_std': float(np.std(hist_b.astype(np.float32).ravel())),
                'red_skew': float(self.calculate_skewness(hist_r)),  # NOUVEAU
                'green_skew': float(self.calculate_skewness(hist_g)),  # NOUVEAU
                'blue_skew': float(self.calculate_skewness(hist_b))   # NOUVEAU
            }
            
            # 2. Contraste adaptatif
            contrast = float(np.std(gray.astype(np.float32).ravel()))
            
            # 3. Détection de contours multi-échelle
            edges_50_150 = cv2.Canny(gray, 50, 150)
            edges_30_100 = cv2.Canny(gray, 30, 100)
            edge_density = float((np.sum(edges_50_150 > 0) + np.sum(edges_30_100 > 0)) / (2 * gray.shape[0] * gray.shape[1]))
            
            # 4. Caractéristiques de texture améliorées
            texture_features = self.calculate_texture_features_advanced(gray)
            
            # 5. Caractéristiques de forme
            shape_features = self.calculate_shape_features(edges_50_150)
            
            # 6. Diversité des couleurs améliorée
            color_diversity = self.calculate_color_diversity_advanced(cv_image)
            
            # 7. Saturation et teinte améliorées
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            saturation = float(np.mean(hsv[:, :, 1].astype(np.float32).ravel()))
            hue_dominance = float(np.std(hsv[:, :, 0].astype(np.float32).ravel()))
            
            return {
                'contrast': contrast,
                'edge_density': edge_density,
                'histogram': hist_features,
                'texture': texture_features,
                'shape': shape_features,
                'color_diversity': color_diversity,
                'saturation': saturation,
                'hue_dominance': hue_dominance
            }
            
        except Exception as e:
            print(f"Erreur dans l'extraction avancée: {e}")
            return {
                'contrast': 30.0,
                'edge_density': 0.1,
                'histogram': {'red_mean': 0, 'green_mean': 0, 'blue_mean': 0, 'red_std': 0, 'green_std': 0, 'blue_std': 0, 'red_skew': 0, 'green_skew': 0, 'blue_skew': 0},
                'texture': {'energy': 0, 'entropy': 0, 'contrast': 0, 'homogeneity': 0},
                'shape': {'area_ratio': 0, 'perimeter_ratio': 0, 'circularity': 0},
                'color_diversity': 0.0,
                'saturation': 0.0,
                'hue_dominance': 0.0
            }
    
    def calculate_skewness(self, histogram):
        """Calculer l'asymétrie d'un histograme"""
        try:
            hist_flat = histogram.flatten()
            mean = np.mean(hist_flat)
            std = np.std(hist_flat)
            if std == 0:
                return 0
            skewness = np.mean(((hist_flat - mean) / std) ** 3)
            return skewness
        except:
            return 0
    
    def calculate_texture_features_advanced(self, gray):
        """Calculer les caractéristiques de texture avec des méthodes avancées"""
        try:
            # Matrice de cooccurrence simplifiée
            glcm = self.calculate_glcm(gray)
            
            # Énergie
            energy = float(np.sum(glcm ** 2))
            
            # Entropie
            entropy = float(-np.sum(glcm * np.log(glcm + 1e-10)))
            
            # Contraste local
            contrast = float(np.sum(glcm * (np.arange(256)[:, None] - np.arange(256)[None, :]) ** 2))
            
            # Homogénéité
            homogeneity = float(np.sum(glcm / (1 + (np.arange(256)[:, None] - np.arange(256)[None, :]) ** 2)))
            
            return {
                'energy': energy,
                'entropy': entropy,
                'contrast': contrast,
                'homogeneity': homogeneity
            }
        except:
            return {'energy': 0, 'entropy': 0, 'contrast': 0, 'homogeneity': 0}
    
    def calculate_glcm(self, gray):
        """Calculer une matrice de cooccurrence simplifiée"""
        try:
            # Réduire la résolution pour la performance
            small_gray = cv2.resize(gray, (64, 64))
            glcm = np.zeros((256, 256))
            
            for i in range(small_gray.shape[0] - 1):
                for j in range(small_gray.shape[1] - 1):
                    current_pixel = small_gray[i, j]
                    next_pixel = small_gray[i, j + 1]
                    glcm[current_pixel, next_pixel] += 1
            
            # Normaliser
            glcm = glcm / np.sum(glcm)
            return glcm
        except:
            return np.ones((256, 256)) / (256 * 256)
    
    def calculate_shape_features(self, edges):
        """Calculer les caractéristiques de forme"""
        try:
            # Trouver les contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return {'area_ratio': 0, 'perimeter_ratio': 0, 'circularity': 0}
            
            # Prendre le plus grand contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Aire et périmètre
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Ratios par rapport à l'image
            image_area = edges.shape[0] * edges.shape[1]
            area_ratio = area / image_area
            perimeter_ratio = perimeter / (2 * (edges.shape[0] + edges.shape[1]))
            
            # Circularité
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            return {
                'area_ratio': float(area_ratio),
                'perimeter_ratio': float(perimeter_ratio),
                'circularity': float(circularity)
            }
        except:
            return {'area_ratio': 0, 'perimeter_ratio': 0, 'circularity': 0}
    
    def calculate_color_diversity(self, image):
        """Calculer la diversité des couleurs"""
        try:
            # Convertir en RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Réduire la résolution pour la performance
            small_image = cv2.resize(rgb_image, (32, 32))
            
            # Compter les couleurs uniques
            pixels = small_image.reshape(-1, 3)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1])))))
            
            # Normaliser par le nombre total de pixels
            diversity = unique_colors / (32 * 32)
            
            return diversity
        except:
            return 0.0
    
    def calculate_color_diversity_advanced(self, image):
        """Calculer la diversité des couleurs de manière plus sophistiquée"""
        try:
            # Convertir en RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Quantifier les couleurs en groupes
            small_image = cv2.resize(rgb_image, (64, 64))
            pixels = small_image.reshape(-1, 3)
            
            # Utiliser une approximation de k-means simple
            # Compter les couleurs uniques après réduction de la palette
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1])))))
            
            # Calculer l'entropie approximative
            hist, _ = np.histogramdd(pixels, bins=[16, 16, 16])
            hist = hist.flatten()
            hist = hist[hist > 0]  # Supprimer les zéros
            
            # Calculer l'entropie de Shannon
            probabilities = hist / hist.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Normaliser par l'entropie maximale possible
            max_entropy = np.log2(len(hist))
            diversity = entropy / max_entropy if max_entropy > 0 else 0
            
            return min(diversity, 1.0)  # S'assurer que c'est entre 0 et 1
        except:
            # Fallback vers l'ancienne méthode
            return self.calculate_color_diversity(image)

class SimpleClassifier:
    def __init__(self):
        """Initialiser le classificateur simple ou IA"""
        self.rules = self.load_classification_rules()
        self.ia_model = None
        self.ia_trained = False
        self.ia_label_map = {0: 'dirty', 1: 'clean'}
        self.mode = 'rules'  # 'rules' or 'ia'
        print("✓ Classificateur simple initialisé")

    def set_mode(self, mode):
        """Changer le mode de classification ('rules' ou 'ia')"""
        if mode in ['rules', 'ia']:
            self.mode = mode
        else:
            print(f"Mode inconnu: {mode}")

    def classify(self, features):
        """Classifier une image selon le mode choisi"""
        if self.mode == 'ia':
            return self.classify_ia(features)
        else:
            return self.classify_rules(features)

    def classify_rules(self, features):
        """Ancienne méthode de classification par règles"""
        # Copie du code original de classify ici
        try:
            if not self.rules:
                self.rules = self.load_classification_rules()
            for rule in self.rules:
                if self.evaluate_rule(features, rule['condition']):
                    confidence = self.calculate_confidence(features, rule['condition'])
                    action = rule['action']
                    if action == 'full':
                        return 'pleine', confidence
                    elif action == 'empty':
                        return 'vide', confidence
                    else:
                        return action, confidence
            file_size = features.get('file_size', 0)
            width = features.get('width', 0)
            height = features.get('height', 0)
            brightness = features.get('brightness', 128)
            unique_hash = abs(int(file_size + width * height + brightness * 100))
            if unique_hash % 2 == 0:
                return 'vide', 0.75
            else:
                return 'pleine', 0.75
        except Exception as e:
            print(f"⚠️ Erreur classification: {e}")
            import random
            return ('vide', 0.6) if random.randint(0, 1) == 0 else ('pleine', 0.6)

    def classify_ia(self, features):
        """Classifier une image avec un modèle IA entraîné sur les données labellisées"""
        if not self.ia_trained:
            print("Le modèle IA n'est pas entraîné. Veuillez le réentraîner via l'admin.")
            return 'inconnu', 0.0
        if self.ia_model is None:
            print("Aucun modèle IA disponible")
            return 'inconnu', 0.0
        # Sélectionner les features utilisés pour l'IA
        X = self.features_to_vector(features)
        pred = self.ia_model.predict([X])[0]
        proba = max(self.ia_model.predict_proba([X])[0])
        label = self.ia_label_map.get(pred, 'inconnu')
        # Map to 'pleine'/'vide' for compatibility
        if label == 'dirty':
            return 'pleine', proba
        elif label == 'clean':
            return 'vide', proba
        else:
            return label, proba

    def train_ia_model(self):
        """Entraîner le modèle IA sur les images labellisées"""
        print("Entraînement du modèle IA...")
        X, y = [], []
        for label, folder in [(0, 'dirty'), (1, 'clean')]:
            dir_path = os.path.join('Data', 'train', 'with_label', folder)
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                try:
                    feats = feature_extractor.extract(fpath)
                    if feats:
                        X.append(self.features_to_vector(feats))
                        y.append(label)
                except Exception as e:
                    print(f"Erreur extraction IA: {e}")
        if len(X) < 5:
            print("Pas assez de données pour entraîner l'IA")
            self.ia_model = None
            self.ia_trained = False
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Score IA (test): {acc:.2f}")
        y_pred = model.predict(X_test)
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        self.ia_model = model
        self.ia_trained = True
        # Optionally save model: joblib.dump(model, 'ia_model.joblib')
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation (5-fold) accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    def features_to_vector(self, features):
        """Convertir le dict de features en vecteur pour le modèle IA"""
        # Sélectionner les features numériques pertinents
        vec = [
            features.get('file_size', 0),
            features.get('width', 0),
            features.get('height', 0),
            features.get('avg_red', 0),
            features.get('avg_green', 0),
            features.get('avg_blue', 0),
            features.get('brightness', 0),
            features.get('contrast_level', 0),
            features.get('edge_density', 0),
            features.get('color_diversity', 0),
            features.get('saturation', 0),
            features.get('hue_dominance', 0)
        ]
        # Optionally add more from histogram/texture/shape
        return vec

    def load_classification_rules(self):
        """Charger les règles de classification depuis la base de données"""
        try:
            conn = sqlite3.connect('binsight.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT condition_json, action, priority 
                FROM classification_rules 
                WHERE active = 1 
                ORDER BY priority ASC
            ''')
            
            rules = []
            for row in cursor.fetchall():
                try:
                    condition = json.loads(row[0])
                    rules.append({
                        'condition': condition,
                        'action': row[1],
                        'priority': row[2]
                    })
                except json.JSONDecodeError:
                    print(f"⚠️ Règle ignorée - JSON invalide: {row[0]}")
            
            conn.close()
            print(f"✓ {len(rules)} règles de classification chargées")
            return rules
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des règles: {e}")
            return []
    
    def evaluate_rule(self, features, condition):
        """Évaluer si une condition est remplie"""
        try:
            for feature_path, rule in condition.items():
                value = self.get_nested_value(features, feature_path)
                operator = rule.get('operator', '==')
                threshold = rule.get('value', 0)
                
                if not self.compare_values(value, operator, threshold):
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Erreur évaluation règle: {e}")
            return False
    
    def get_nested_value(self, data, path):
        """Récupérer une valeur dans un dictionnaire imbriqué"""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return 0  # Valeur par défaut si la clé n'existe pas
            return value
        except:
            return 0
    
    def compare_values(self, value, operator, threshold):
        """Comparer deux valeurs selon l'opérateur"""
        try:
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            elif operator == '!=':
                return value != threshold
            else:
                return False
        except:
            return False
    
    def calculate_confidence(self, features, condition):
        """Calculer un niveau de confiance pour la classification"""
        try:
            # Logique simple pour calculer la confiance
            brightness = features.get('brightness', 128)
            contrast = features.get('contrast_level', 50)
            
            # Plus les valeurs sont extrêmes, plus la confiance est haute
            brightness_confidence = abs(brightness - 128) / 128
            contrast_confidence = min(contrast / 100, 1.0)
            
            return min(max((brightness_confidence + contrast_confidence) / 2, 0.5), 0.95)
            
        except:
            return 0.6

# Initialiser les composants
feature_extractor = FeatureExtractor()
classifier = SimpleClassifier()
classifier.train_ia_model()  # Entraîne l'IA au démarrage

# Middleware pour logger les requêtes
@app.before_request
def log_request_info():
    if request.endpoint not in ['uploaded_file', 'index']:  # Éviter spam pour les ressources statiques
        print(f"=== REQUÊTE {request.method} {request.path} ===")
        if request.method == 'POST':
            print(f"Files: {list(request.files.keys())}")
            print(f"Form keys: {list(request.form.keys())}")

@app.route('/')
def index():
    return render_template('BinSight.html')

@app.route('/test')
def test():
    """Route de test pour vérifier la connexion"""
    return jsonify({
        'status': 'OK', 
        'message': 'Serveur Flask fonctionne',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-db')
def test_db():
    """Tester la structure de la base de données"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(images)")
    columns = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) FROM images")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    column_names = [col[1] for col in columns]
    
    return jsonify({
        'columns': column_names,
        'has_user_annotation': 'user_annotation' in column_names,
        'total_columns': len(column_names),
        'total_images': count,
        'status': 'OK'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gérer l'upload d'images avec débogage complet"""
    try:
        print("=== DÉBUT UPLOAD ===")
        
        if 'file' not in request.files:
            print("❌ Erreur: Aucun fichier dans la requête")
            return jsonify({'success': False, 'message': 'Aucun fichier trouvé dans la requête'})
        
        file = request.files['file']
        print(f"📁 Fichier reçu: {file.filename}")
        
        if file.filename is None:
            print("❌ Erreur: Nom de fichier est None")
            return jsonify({'success': False, 'message': 'Nom de fichier manquant'})
        
        if file.filename == '':
            print("❌ Erreur: Nom de fichier vide")
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'})
        
        # Récupérer les données du formulaire
        location = request.form.get('location', '').strip()
        bin_type = request.form.get('bin_type', '').strip()
        comment = request.form.get('comment', '').strip()
        user_annotation = request.form.get('user_annotation', '').strip()

        
        p1 = (-30.250704671553954, -57.14852536982011)
        p2 = (-32.89257311452824, -53.43179943666392)
        p3 = (-34.88099515435342, -54.89218198697705)
        p4 = (-34.11095493205359, -58.14044352690463)

        latitudes = [p1[0], p2[0], p3[0], p4[0]]
        longitudes = [p1[1], p2[1], p3[1], p4[1]]

        min_latitude = min(latitudes)
        max_latitude = max(latitudes)
        min_longitude = min(longitudes)
        max_longitude = max(longitudes)

        random_latitude = random.uniform(min_latitude, max_latitude)
        random_longitude = random.uniform(min_longitude, max_longitude)

        location = str((random_latitude, random_longitude))

        print(location)
        
        print(f"📝 Données form:")
        print(f"   - Location: '{location}'")
        print(f"   - Type: '{bin_type}'")
        print(f"   - Annotation: '{user_annotation}'")
        print(f"   - Comment: '{comment}'")
        
        if file and allowed_file(file.filename):
            print("✅ Fichier valide, début sauvegarde...")
            
            # Sauvegarder le fichier
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"💾 Chemin de sauvegarde: {file_path}")
            
            # Sauvegarder
            file.save(file_path)
            file_exists = os.path.exists(file_path)
            print(f"✅ Fichier sauvé: {file_exists}")
            
            if not file_exists:
                return jsonify({'success': False, 'message': 'Erreur lors de la sauvegarde du fichier'})
            
            # Extraire les caractéristiques
            print("🔍 Début extraction des caractéristiques...")
            # MODIFIÉ: Correction du nom de la méthode
            features = feature_extractor.extract(file_path)
            
            if features:
                print("✅ Caractéristiques extraites avec succès")
                
                # Déterminer la classification
                if user_annotation and user_annotation in ['pleine', 'vide']:
                    final_classification = user_annotation
                    confidence = 1.0
                    message = f'Annotation utilisateur: Poubelle {user_annotation}'
                    is_user_annotation = True
                    auto_classification = None
                    print(f"👤 Classification utilisateur: {final_classification}")
                else:
                    auto_classification, confidence = classifier.classify(features)
                    final_classification = auto_classification
                    message = f'Analyse IA: Poubelle {auto_classification} détectée'
                    is_user_annotation = False
                    print(f"🤖 Classification IA: {final_classification} (confiance: {confidence:.2f})")
                
                # Sauvegarder en base de données
                print("💾 Sauvegarde en base de données...")
                conn = sqlite3.connect('binsight.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO images (
                        filename, file_path, file_size, width, height,
                        avg_red, avg_green, avg_blue, contrast_level,
                        brightness, edge_density, location, bin_type, 
                        comment, auto_classification, user_annotation, confidence,
                        histogram_features, texture_features, shape_features,
                        color_diversity, saturation, hue_dominance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename, file_path, features['file_size'], features['width'], 
                    features['height'], features['avg_red'], features['avg_green'], 
                    features['avg_blue'], features['contrast_level'], features['brightness'],
                    features['edge_density'], location if location else None, 
                    bin_type if bin_type else None, comment if comment else None, 
                    auto_classification, user_annotation if user_annotation else None, confidence,
                    json.dumps(features['histogram_features']),
                    json.dumps(features['texture_features']),
                    json.dumps(features['shape_features']),
                    features['color_diversity'], features['saturation'], features['hue_dominance']
                ))
                
                image_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                print(f"✅ Sauvé en BDD avec ID: {image_id}")
                
                response_data = {
                    'success': True,
                    'message': message,
                    'classification': final_classification,
                    'confidence': f'{confidence*100:.1f}%',
                    'features': features,
                    'image_id': image_id,
                    'image_url': url_for('uploaded_file', filename=filename),
                    'is_user_annotation': is_user_annotation
                }
                
                print(f"✅ Upload terminé avec succès")
                return jsonify(response_data)
                
            else:
                print("❌ Erreur: Échec extraction des caractéristiques")
                return jsonify({'success': False, 'message': 'Erreur lors de l\'analyse de l\'image'})
        else:
            extension = file.filename.split('.')[-1] if file.filename and '.' in file.filename else 'aucune'
            print(f"❌ Fichier non valide. Extension: {extension}")
            return jsonify({'success': False, 'message': 'Format de fichier non supporté. Utilisez JPG, PNG ou JPEG.'})
    
    except Exception as e:
        print(f"❌ ERREUR GÉNÉRALE: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Erreur serveur: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir les images uploadées"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/annotate/<int:image_id>/<annotation>')
def annotate_image(image_id, annotation):
    """Annotation manuelle d'une image"""
    if annotation in ['pleine', 'vide']:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE images SET manual_annotation = ? WHERE id = ?', (annotation, image_id))
        conn.commit()
        conn.close()
        print(f"✅ Image {image_id} annotée manuellement comme {annotation}")
        return jsonify({'success': True, 'message': f'Image annotée manuellement comme {annotation}'})
    return jsonify({'success': False, 'message': 'Annotation invalide'})

@app.route('/api/stats')
def api_stats():
    """API pour les statistiques"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE date(upload_date) = date("now")')
        today_uploads = cursor.fetchone()[0]
        
        cursor.execute('''SELECT COUNT(*) FROM images WHERE 
            user_annotation = "pleine" OR 
            (user_annotation IS NULL AND manual_annotation = "pleine") OR 
            (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = "pleine")''')
        full_bins = cursor.fetchone()[0]
        
        cursor.execute('''SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN manual_annotation = auto_classification THEN 1 ELSE 0 END) as correct
            FROM images WHERE manual_annotation IS NOT NULL AND user_annotation IS NULL AND auto_classification IS NOT NULL''')
        precision_data = cursor.fetchone()
        accuracy = 0.0
        if precision_data[0] > 0:
            accuracy = (precision_data[1] / precision_data[0]) * 100
        
        cursor.execute('''SELECT 
            strftime("%m", upload_date) as month,
            COUNT(*) as total,
            SUM(CASE WHEN 
                user_annotation = "vide" OR 
                (user_annotation IS NULL AND manual_annotation = "vide") OR 
                (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = "vide") 
                THEN 1 ELSE 0 END) as empty,
            SUM(CASE WHEN 
                user_annotation = "pleine" OR 
                (user_annotation IS NULL AND manual_annotation = "pleine") OR 
                (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = "pleine") 
                THEN 1 ELSE 0 END) as full
            FROM images 
            WHERE strftime("%Y", upload_date) = strftime("%Y", "now")
            GROUP BY strftime("%m", upload_date)
            ORDER BY month''')
        monthly_data = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'today_uploads': today_uploads,
            'full_bins': full_bins,
            'accuracy': round(accuracy, 1),
            'monthly_data': monthly_data
        })
        
    except Exception as e:
        print(f"Erreur API stats: {e}")
        return jsonify({
            'today_uploads': 0,
            'full_bins': 0,
            'accuracy': 0.0,
            'monthly_data': []
        })

@app.route('/api/recent_reports')
def get_recent_reports():
    """
    Récupère les signalements récents avec pagination.
    Accepte les paramètres `page` et `limit` dans l'URL.
    """
    print("=== REQUÊTE GET /api/recent_reports ===")
    
    # NOUVEAU: Gérer la pagination
    page = request.args.get('page', 1, type=int)
    limit = 10  # Nombre de signalements par page
    offset = (page - 1) * limit

    conn = sqlite3.connect('binsight.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # NOUVEAU: Compter le nombre total de signalements pour la pagination
        cursor.execute('SELECT COUNT(id) FROM images')
        total_reports = cursor.fetchone()[0]
        total_pages = (total_reports + limit - 1) // limit

        # MODIFIÉ: Récupérer les signalements pour la page actuelle
        cursor.execute('SELECT * FROM images ORDER BY upload_date DESC LIMIT ? OFFSET ?', (limit, offset))
        images = cursor.fetchall()
        
        reports_list = []
        for image in images:
            image_dict = dict(image)
            # Le statut est l'annotation de l'utilisateur, sinon celle manuelle, sinon celle de l'IA
            status = image_dict.get('user_annotation') or image_dict.get('manual_annotation') or image_dict.get('auto_classification') or 'inconnu'
            
            # MODIFIÉ: Utiliser la nouvelle fonction de parsing
            upload_date_obj = parse_db_timestamp(image_dict['upload_date'])

            reports_list.append({
                'id': image_dict['id'],
                'filename': image_dict['filename'],
                'image_url': url_for('static', filename=f'uploads/{image_dict["filename"]}'),
                'location': image_dict['location'],
                'bin_type': image_dict['bin_type'],
                'date': upload_date_obj.strftime('%d/%m/%Y %H:%M') if upload_date_obj else 'Date inconnue',
                'status': status
            })
        
        # NOUVEAU: Retourner les données avec les informations de pagination
        return jsonify({
            'reports': reports_list,
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_reports': total_reports,
                'limit': limit
            }
        })

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des signalements: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        conn.close()


@app.route('/api/map_data')
def get_map_data():
    """
    NOUVEAU: Fournit les données pour la cartographie.
    """
    print("=== REQUÊTE GET /api/map_data ===")
    conn = sqlite3.connect('binsight.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Sélectionner les images qui ont une localisation valide
        cursor.execute("SELECT id, location, upload_date, user_annotation, manual_annotation, auto_classification, confidence FROM images WHERE location IS NOT NULL AND location != ''")
        images = cursor.fetchall()
        
        map_data = []
        for image in images:
            image_dict = dict(image)
            location_str = image_dict.get('location')
            
            lat, lng = None, None
            try:
                # Tenter de parser les coordonnées (ex: "48.85,2.35")
                if location_str:
                    parts = location_str.replace('(', '').replace(')', '').split(',')
                    if len(parts) == 2:
                        lat = float(parts[0].strip())
                        lng = float(parts[1].strip())
            except (ValueError, TypeError):
                # Ignorer cette entrée si les coordonnées sont invalides
                continue

            if lat is not None and lng is not None:
                status = image_dict.get('user_annotation') or image_dict.get('manual_annotation') or image_dict.get('auto_classification') or 'inconnu'
                upload_date_obj = parse_db_timestamp(image_dict['upload_date'])

                map_data.append({
                    'id': image_dict['id'],
                    'lat': lat,
                    'lng': lng,
                    'status': status,
                    'time': upload_date_obj.strftime('%d/%m %H:%M') if upload_date_obj else 'N/A',
                    'location': location_str,
                    'confidence': image_dict.get('confidence')
                })
        
        print(f"✅ {len(map_data)} points chargés pour la carte.")
        return jsonify(map_data)

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données pour la carte: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        conn.close()


@app.route('/api/image/<int:image_id>')
def get_image_details(image_id):
    """Récupérer les détails d'une image"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
        image = cursor.fetchone()
        conn.close()
        
        if image:
            columns = ['id', 'filename', 'file_path', 'upload_date', 'manual_annotation', 
                      'auto_classification', 'user_annotation', 'file_size', 'width', 'height', 'avg_red', 
                      'avg_green', 'avg_blue', 'contrast_level', 'brightness', 'edge_density', 
                      'location', 'bin_type', 'comment', 'confidence']
            
            image_dict = dict(zip(columns, image))
            image_dict['image_url'] = url_for('uploaded_file', filename=image[1])
            
            return jsonify(image_dict)
        
        return jsonify({'error': 'Image non trouvée'}), 404
        
    except Exception as e:
        print(f"Erreur API image details: {e}")
        return jsonify({'error': 'Erreur serveur'}), 500

@app.route('/api/rules', methods=['GET'])
def get_classification_rules():
    """Récupérer toutes les règles de classification"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, description, condition_json, action, priority, active, created_date
        FROM classification_rules
        ORDER BY priority ASC, created_date DESC
    ''')
    
    rules = []
    for row in cursor.fetchall():
        rules.append({
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'condition_json': row[3],
            'action': row[4],
            'priority': row[5],
            'active': row[6],
            'created_date': row[7]
        })
    
    conn.close()
    return jsonify(rules)

@app.route('/api/rules', methods=['POST'])
def create_classification_rule():
    """Créer une nouvelle règle de classification"""
    data = request.get_json()
    
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO classification_rules (name, description, condition_json, action, priority, active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['name'],
            data.get('description', ''),
            data['condition_json'],
            data['action'],
            data.get('priority', 1),
            data.get('active', True)
        ))
        
        conn.commit()
        rule_id = cursor.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'rule_id': rule_id})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rules/<int:rule_id>', methods=['PUT'])
def update_classification_rule(rule_id):
    """Mettre à jour une règle de classification"""
    data = request.get_json()
    
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE classification_rules 
            SET name = ?, description = ?, condition_json = ?, action = ?, priority = ?, active = ?, modified_date = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            data['name'],
            data.get('description', ''),
            data['condition_json'],
            data['action'],
            data.get('priority', 1),
            data.get('active', True),
            rule_id
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rules/<int:rule_id>', methods=['DELETE'])
def delete_classification_rule(rule_id):
    """Supprimer une règle de classification"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM classification_rules WHERE id = ?', (rule_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/rules')
def rules_management():
    """Page de gestion des règles de classification"""
    return render_template('rules.html')

@app.route('/api/images/<int:image_id>/metadata', methods=['POST'])
def save_annotation_metadata(image_id):
    """Sauvegarder les métadonnées d'annotation"""
    data = request.get_json()
    
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO annotation_metadata (image_id, annotation_type, metadata_json)
            VALUES (?, ?, ?)
        ''', (image_id, data['type'], json.dumps(data)))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/images/<int:image_id>/annotate', methods=['POST'])
def detailed_annotate(image_id):
    """Annotation détaillée d'une image"""
    data = request.get_json()
    
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE images 
            SET manual_annotation = ?, bin_type = ?, location = ?, comment = ?
            WHERE id = ?
        ''', (
            data['annotation'],
            data.get('bin_type', ''),
            data.get('location', ''),
            data.get('comment', ''),
            image_id
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'annotation': data['annotation']})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/annotate/<int:image_id>')
def annotate_interface(image_id):
    """Interface d'annotation avancée"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    # Get current image
    cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
    current_image = cursor.fetchone()
    
    if not current_image:
        return "Image not found", 404
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM images')
    total_images = cursor.fetchone()[0]
    
    # Get previous/next images
    cursor.execute('SELECT id FROM images WHERE id < ? ORDER BY id DESC LIMIT 1', (image_id,))
    prev_result = cursor.fetchone()
    previous_image = {'id': prev_result[0]} if prev_result else None
    
    cursor.execute('SELECT id FROM images WHERE id > ? ORDER BY id ASC LIMIT 1', (image_id,))
    next_result = cursor.fetchone()
    next_image = {'id': next_result[0]} if next_result else None
    
    conn.close()
    
    # Convert to dict
    columns = ['id', 'filename', 'file_path', 'upload_date', 'manual_annotation', 
               'auto_classification', 'user_annotation', 'file_size', 'width', 'height', 
               'avg_red', 'avg_green', 'avg_blue', 'contrast_level', 'brightness', 
               'edge_density', 'location', 'bin_type', 'comment', 'confidence',
               'histogram_features', 'texture_features', 'shape_features', 
               'color_diversity', 'saturation', 'hue_dominance']
    
    current_image_dict = dict(zip(columns, current_image))
    
    return render_template('annotate.html', 
                         current_image=current_image_dict,
                         total_images=total_images,
                         previous_image=previous_image,
                         next_image=next_image)

@app.route('/dashboard')
def dashboard():
    """Tableau de bord interactif"""
    return render_template('dashboard.html')

@app.route('/api/dashboard/metrics')
def dashboard_metrics():
    """API pour les métriques du dashboard"""
    try:
        period = request.args.get('period', '30')
        bin_type = request.args.get('bin_type', 'all')
        location = request.args.get('location', 'all')
        
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        # Base query with filters
        where_conditions = [f"upload_date >= datetime('now', '-{period} days')"]
        params = []
        
        if bin_type != 'all':
            where_conditions.append("bin_type = ?")
            params.append(bin_type)
            
        if location != 'all':
            where_conditions.append("location LIKE ?")
            params.append(f"%{location}%")
        
        where_clause = " AND ".join(where_conditions)
        
        # Total images
        cursor.execute(f"SELECT COUNT(*) FROM images WHERE {where_clause}", params)
        total_images = cursor.fetchone()[0]
        
        # Full bins
        full_condition = "(user_annotation = 'pleine' OR (user_annotation IS NULL AND manual_annotation = 'pleine') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'pleine'))"
        cursor.execute(f"SELECT COUNT(*) FROM images WHERE {where_clause} AND {full_condition}", params)
        full_bins = cursor.fetchone()[0]
        
        # Empty bins
        empty_condition = "(user_annotation = 'vide' OR (user_annotation IS NULL AND manual_annotation = 'vide') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'vide'))"
        cursor.execute(f"SELECT COUNT(*) FROM images WHERE {where_clause} AND {empty_condition}", params)
        empty_bins = cursor.fetchone()[0]
        
        # Accuracy
        cursor.execute(f"""SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN manual_annotation = auto_classification THEN 1 ELSE 0 END) as correct
            FROM images WHERE {where_clause} AND manual_annotation IS NOT NULL AND auto_classification IS NOT NULL""", params)
        
        accuracy_data = cursor.fetchone()
        accuracy = 0.0
        if accuracy_data[0] > 0:
            accuracy = (accuracy_data[1] / accuracy_data[0])
        
        conn.close()
        
        return jsonify({
            'total_images': total_images,
            'full_bins': full_bins,
            'empty_bins': empty_bins,
            'accuracy': accuracy
        })
        
    except Exception as e:
        print(f"Erreur dashboard metrics: {e}")
        return jsonify({'total_images': 0, 'full_bins': 0, 'empty_bins': 0, 'accuracy': 0.0})

@app.route('/api/dashboard/timeline')
def dashboard_timeline():
    """Données temporelles pour le dashboard"""
    try:
        period = int(request.args.get('period', '30'))
        
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT 
                date(upload_date) as day,
                COUNT(*) as total,
                SUM(CASE WHEN (user_annotation = 'vide' OR (user_annotation IS NULL AND manual_annotation = 'vide') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'vide')) THEN 1 ELSE 0 END) as empty,
                SUM(CASE WHEN (user_annotation = 'pleine' OR (user_annotation IS NULL AND manual_annotation = 'pleine') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'pleine')) THEN 1 ELSE 0 END) as full
            FROM images 
            WHERE upload_date >= datetime('now', '-{period} days')
            GROUP BY date(upload_date)
            ORDER BY day
        """)
        
        data = cursor.fetchall()
        conn.close()
        
        labels = [row[0] for row in data]
        empty_data = [row[2] for row in data]
        full_data = [row[3] for row in data]
        
        return jsonify({
            'labels': labels,
            'empty_data': empty_data,
            'full_data': full_data
        })
        
    except Exception as e:
        print(f"Erreur dashboard timeline: {e}")
        return jsonify({'labels': [], 'empty_data': [], 'full_data': []})

@app.route('/api/dashboard/status')
def dashboard_status():
    """Répartition des statuts"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        # Count each status
        cursor.execute("SELECT COUNT(*) FROM images WHERE (user_annotation = 'pleine' OR (user_annotation IS NULL AND manual_annotation = 'pleine') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'pleine'))")
        full = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM images WHERE (user_annotation = 'vide' OR (user_annotation IS NULL AND manual_annotation = 'vide') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'vide'))")
        empty = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM images WHERE (user_annotation = 'half' OR (user_annotation IS NULL AND manual_annotation = 'half') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'half'))")
        half = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM images WHERE (user_annotation = 'unclear' OR (user_annotation IS NULL AND manual_annotation = 'unclear') OR (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'unclear'))")
        unclear = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'full': full,
            'empty': empty,
            'half': half,
            'unclear': unclear
        })
        
    except Exception as e:
        print(f"Erreur dashboard status: {e}")
        return jsonify({'full': 0, 'empty': 0, 'half': 0, 'unclear': 0})

@app.route('/api/locations')
def api_locations():
    """Liste des emplacements"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT location FROM images WHERE location IS NOT NULL AND location != ''")
        locations = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return jsonify(locations)
        
    except Exception as e:
        print(f"Erreur API locations: {e}")
        return jsonify([])

# 🔧 FONCTION DE TEST
@app.route('/test-upload')
def test_upload():
    """Route de test pour vérifier l'upload d'images"""
    test_image_path = 'static/test_image.jpg'  # Chemin vers une image de test sur le serveur
    
    if not os.path.exists(test_image_path):
        return "Image de test non trouvée", 404
    
    # Simuler une requête d'upload
    with app.test_request_context('/upload', method='POST', 
                                   data={'file': (open(test_image_path, 'rb'), 'test_image.jpg')}):
        response = upload_file()
        return response

@app.route('/api/verify_integrity')
def verify_data_integrity():
    """
    Vérifie l'intégrité des données entre la base de données et les fichiers.
    """
    print("🕵️ Démarrage de la vérification de l'intégrité des données...")
    report = {
        'missing_files': [],
        'orphaned_files': [],
        'invalid_json_features': [],
        'invalid_confidence': [],
        'summary': {}
    }
    
    conn = sqlite3.connect('binsight.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Vérifier les fichiers manquants et les données invalides
    cursor.execute('SELECT id, file_path, confidence, histogram_features, texture_features, shape_features FROM images')
    db_images = cursor.fetchall()
    db_filepaths = set()

    for image in db_images:
        db_filepaths.add(image['file_path'])
        
        # Vérification de l'existence du fichier
        if not os.path.exists(image['file_path']):
            report['missing_files'].append({'id': image['id'], 'path': image['file_path']})
            
        # Vérification de la confiance
        if image['confidence'] is not None and not (0 <= image['confidence'] <= 1):
            report['invalid_confidence'].append({'id': image['id'], 'confidence': image['confidence']})
            
        # Vérification des JSON de caractéristiques
        for feature_col in ['histogram_features', 'texture_features', 'shape_features']:
            try:
                if image[feature_col]:
                    json.loads(image[feature_col])
            except (json.JSONDecodeError, TypeError):
                report['invalid_json_features'].append({'id': image['id'], 'column': feature_col})
                break # Inutile de vérifier les autres colonnes JSON pour cette image

    # 2. Vérifier les fichiers orphelins
    upload_folder = app.config['UPLOAD_FOLDER']
    disk_files = set()
    for filename in os.listdir(upload_folder):
        # Ignorer les sous-dossiers comme 'thumbnails' s'il y en a
        full_path = os.path.join(upload_folder, filename)
        if os.path.isfile(full_path):
            disk_files.add(full_path.replace('\\', '/'))
            
    orphaned_paths = disk_files - db_filepaths
    report['orphaned_files'] = list(orphaned_paths)
    
    conn.close()
    
    # 3. Générer un résumé
    report['summary'] = {
        'total_db_records': len(db_images),
        'total_disk_files': len(disk_files),
        'missing_files_count': len(report['missing_files']),
        'orphaned_files_count': len(report['orphaned_files']),
        'invalid_json_count': len(report['invalid_json_features']),
        'invalid_confidence_count': len(report['invalid_confidence']),
    }
    
    print(f"✅ Vérification terminée. Rapport: {report['summary']}")
    return jsonify(report)


@app.route('/api/batch_import', methods=['POST'])
def batch_import_images():
    """
    MODIFIÉ: Importe en masse les images avec vérification stricte des doublons.
    """
    print("🚀 Démarrage de l'importation en masse...")
    source_dirs = ['Data/test', 'Data/train']
    upload_folder = app.config['UPLOAD_FOLDER']
    
    report = {
        'imported': 0,
        'skipped': 0,
        'failed': 0,
        'failed_files': [],
        'duplicate_details': []  # NOUVEAU: Détails des doublons
    }

    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()

        # MODIFIÉ: Obtenir les noms de fichiers ET chemins existants pour éviter les doublons
        cursor.execute("SELECT filename, file_path FROM images")
        existing_data = cursor.fetchall()
        existing_filenames = {row[0] for row in existing_data}
        existing_paths = {row[1].replace('\\', '/') for row in existing_data}
        
        print(f"ℹ️ {len(existing_filenames)} fichiers déjà présents dans la base de données.")
        print(f"ℹ️ {len(existing_paths)} chemins de fichiers existants.")

        # NOUVEAU: Calculer la taille des fichiers sources avant importation
        total_files_to_process = 0
        files_to_import = []
        
        for source_dir in source_dirs:
            if not os.path.isdir(source_dir):
                print(f"⚠️ Dossier source non trouvé: {source_dir}")
                continue

            for filename in os.listdir(source_dir):
                if not allowed_file(filename):
                    continue
                
                source_path = os.path.join(source_dir, filename)
                secure_name = secure_filename(filename)
                dest_path = os.path.join(upload_folder, secure_name).replace('\\', '/')
                
                # NOUVEAU: Vérifications multiples pour éviter les doublons
                is_duplicate = False
                duplicate_reason = None
                
                # 1. Vérifier par nom de fichier
                if secure_name in existing_filenames:
                    is_duplicate = True
                    duplicate_reason = f"Nom de fichier déjà existant: {secure_name}"
                
                # 2. Vérifier par chemin de destination
                elif dest_path in existing_paths:
                    is_duplicate = True
                    duplicate_reason = f"Chemin de destination déjà utilisé: {dest_path}"
                
                # 3. NOUVEAU: Vérifier si le fichier existe déjà physiquement sur le disque
                elif os.path.exists(dest_path):
                    is_duplicate = True
                    duplicate_reason = f"Fichier déjà présent sur le disque: {dest_path}"
                
                if is_duplicate:
                    report['skipped'] += 1
                    report['duplicate_details'].append({
                        'filename': filename,
                        'reason': duplicate_reason
                    })
                    print(f"⏭️ Ignoré: {duplicate_reason}")
                else:
                    files_to_import.append({
                        'source_path': source_path,
                        'dest_path': dest_path,
                        'secure_name': secure_name,
                        'original_filename': filename
                    })
                
                total_files_to_process += 1

        print(f"📊 Résumé de la pré-analyse:")
        print(f"   - Fichiers trouvés: {total_files_to_process}")
        print(f"   - À importer: {len(files_to_import)}")
        print(f"   - Doublons ignorés: {report['skipped']}")

        # NOUVEAU: Traitement des fichiers validés
        for file_info in files_to_import:
            try:
                print(f"📥 Importation de: {file_info['original_filename']}")
                
                # Copier le fichier
                shutil.copy2(file_info['source_path'], file_info['dest_path'])

                # Extraire les caractéristiques et classifier
                features = feature_extractor.extract(file_info['dest_path'])
                if not features:
                    raise Exception("Échec de l'extraction des caractéristiques")
                
                auto_classification, confidence = classifier.classify(features)

                # Générer des coordonnées GPS aléatoires (Uruguay)
                p1 = (-30.250704671553954, -57.14852536982011)
                p2 = (-32.89257311452824, -53.43179943666392)
                p3 = (-34.88099515435342, -54.89218198697705)
                p4 = (-34.11095493205359, -58.14044352690463)

                latitudes = [p1[0], p2[0], p3[0], p4[0]]
                longitudes = [p1[1], p2[1], p3[1], p4[1]]

                min_latitude = min(latitudes)
                max_latitude = max(latitudes)
                min_longitude = min(longitudes)
                max_longitude = max(longitudes)

                random_latitude = random.uniform(min_latitude, max_latitude)
                random_longitude = random.uniform(min_longitude, max_longitude)
                location = f"{random_latitude:.6f},{random_longitude:.6f}"

                # Insérer dans la base de données
                cursor.execute('''
                    INSERT INTO images (
                        filename, file_path, file_size, width, height,
                        avg_red, avg_green, avg_blue, contrast_level,
                        brightness, edge_density, location, bin_type, 
                        comment, auto_classification, confidence, user_annotation,
                        histogram_features, texture_features, shape_features,
                        color_diversity, saturation, hue_dominance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_info['secure_name'], file_info['dest_path'], features['file_size'], features['width'], 
                    features['height'], features['avg_red'], features['avg_green'], 
                    features['avg_blue'], features['contrast_level'], features['brightness'],
                    features['edge_density'], location, 'Importé', 
                    'Importation en masse', auto_classification, confidence, None,
                    json.dumps(features['histogram_features']),
                    json.dumps(features['texture_features']),
                    json.dumps(features['shape_features']),
                    features['color_diversity'], features['saturation'], features['hue_dominance']
                ))
                
                report['imported'] += 1
                # NOUVEAU: Mettre à jour les listes d'exclusion pour éviter les doublons dans le même lot
                existing_filenames.add(file_info['secure_name'])
                existing_paths.add(file_info['dest_path'])
                
                print(f"✅ Importé avec succès: {file_info['original_filename']}")

            except Exception as e:
                print(f"❌ Erreur lors de l'importation de {file_info['original_filename']}: {e}")
                report['failed'] += 1
                report['failed_files'].append({
                    'filename': file_info['original_filename'],
                    'error': str(e)
                })
                # Supprimer le fichier copié en cas d'erreur
                if os.path.exists(file_info['dest_path']):
                    os.remove(file_info['dest_path'])
        
        conn.commit()
        conn.close()
        
        # NOUVEAU: Rapport détaillé
        print(f"✅ Importation terminée. Rapport détaillé:")
        print(f"   - ✅ Importés: {report['imported']}")
        print(f"   - ⏭️ Ignorés (doublons): {report['skipped']}")
        print(f"   - ❌ Échecs: {report['failed']}")
        
        if report['duplicate_details']:
            print(f"📋 Détails des doublons ignorés:")
            for detail in report['duplicate_details'][:5]:  # Afficher seulement les 5 premiers
                print(f"   - {detail['filename']}: {detail['reason']}")
            if len(report['duplicate_details']) > 5:
                print(f"   - ... et {len(report['duplicate_details']) - 5} autres")
        
        # Diffuser une mise à jour pour que les clients rafraîchissent leurs données
        broadcast_update()

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors de l'importation en masse: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': True, 'report': report})


@app.route('/api/reclassify_all', methods=['POST'])
def reclassify_all_images():
    """
    AMÉLIORÉ: Ré-analyse et re-classifie toutes les images avec détection de doublons.
    """
    print("🔄 Démarrage de la ré-analyse complète avec détection de doublons...")
    report = {
        'reclassified': 0,
        'failed': 0,
        'not_found': 0,
        'duplicates_found': 0,
        'duplicates_removed': 0,
        'failed_files': [],
        'duplicate_details': []
    }

    try:
        conn = sqlite3.connect('binsight.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # NOUVEAU: Détecter les doublons avant la ré-analyse
        print("🔍 Détection des doublons...")
        cursor.execute("""
            SELECT filename, COUNT(*) as count, GROUP_CONCAT(id) as ids
            FROM images 
            GROUP BY filename 
            HAVING count > 1
        """)
        duplicates_by_filename = cursor.fetchall()

        cursor.execute("""
            SELECT file_path, COUNT(*) as count, GROUP_CONCAT(id) as ids
            FROM images 
            GROUP BY file_path 
            HAVING count > 1
        """)
        duplicates_by_path = cursor.fetchall()

        # Supprimer les doublons (garder le plus récent)
        duplicate_ids_to_remove = set()
        
        for dup in duplicates_by_filename:
            ids = [int(x) for x in dup['ids'].split(',')]
            ids_to_remove = ids[:-1]  # Garder le dernier (le plus récent)
            duplicate_ids_to_remove.update(ids_to_remove)
            report['duplicate_details'].append({
                'type': 'filename',
                'value': dup['filename'],
                'count': dup['count'],
                'removed_ids': ids_to_remove
            })

        for dup in duplicates_by_path:
            ids = [int(x) for x in dup['ids'].split(',')]
            ids_to_remove = ids[:-1]  # Garder le dernier
            duplicate_ids_to_remove.update(ids_to_remove)
            report['duplicate_details'].append({
                'type': 'file_path',
                'value': dup['file_path'],
                'count': dup['count'],
                'removed_ids': ids_to_remove
            })

        # MODIFIÉ: Supprimer les doublons de la base de données si nécessaire
        if duplicate_ids_to_remove:
            print(f"🗑️ Suppression de {len(duplicate_ids_to_remove)} doublons...")
            placeholders = ','.join(['?'] * len(duplicate_ids_to_remove))
            cursor.execute(f"DELETE FROM images WHERE id IN ({placeholders})", list(duplicate_ids_to_remove))
            report['duplicates_found'] = len(duplicate_ids_to_remove)
            report['duplicates_removed'] = len(duplicate_ids_to_remove)

        # MODIFIÉ: Obtenir toutes les images restantes pour ré-analyse
        cursor.execute("SELECT id, file_path, filename FROM images")
        all_images = cursor.fetchall()
        print(f"ℹ️ {len(all_images)} images à ré-analyser après nettoyage des doublons.")

        update_cursor = conn.cursor()

        for image in all_images:
            image_id = image['id']
            file_path = image['file_path']
            filename = image['filename']

            if not file_path or not os.path.exists(file_path):
                report['not_found'] += 1
                report['failed_files'].append({'id': image_id, 'reason': 'Fichier non trouvé'})
                continue

            try:
                # Ré-extraire les caractéristiques avec le nouvel extracteur
                features = feature_extractor.extract(file_path)
                if not features:
                    raise Exception("L'extraction des caractéristiques a échoué")

                # Re-classifier avec l'algorithme amélioré
                auto_classification, confidence = classifier.classify(features)

                # Mettre à jour l'enregistrement dans la base de données
                update_cursor.execute('''
                    UPDATE images SET
                        auto_classification = ?,
                        manual_annotation = ?, 
                        confidence = ?,
                        file_size = ?, width = ?, height = ?,
                        avg_red = ?, avg_green = ?, avg_blue = ?,
                        contrast_level = ?, brightness = ?, edge_density = ?,
                        histogram_features = ?, texture_features = ?, shape_features = ?,
                        color_diversity = ?, saturation = ?, hue_dominance = ?
                    WHERE id = ?
                ''', (
                    auto_classification,
                    auto_classification,  # Le nouvel état de l'IA devient l'état de référence
                    confidence,
                    features['file_size'], features['width'], features['height'],
                    features['avg_red'], features['avg_green'], features['avg_blue'],
                    features['contrast_level'], features['brightness'], features['edge_density'],
                    json.dumps(features['histogram_features']),
                    json.dumps(features['texture_features']),
                    json.dumps(features['shape_features']),
                    features['color_diversity'], features['saturation'], features['hue_dominance'],
                    image_id
                ))
                report['reclassified'] += 1

            except Exception as e:
                print(f"❌ Erreur lors de la ré-analyse de {filename} (ID: {image_id}): {e}")
                report['failed'] += 1
                report['failed_files'].append({'id': image_id, 'reason': str(e)})
        
        conn.commit()
        conn.close()
        
        print(f"✅ Ré-analyse terminée. Rapport complet:")
        print(f"   - ✅ Ré-analysées: {report['reclassified']}")
        print(f"   - 🗑️ Doublons supprimés: {report['duplicates_removed']}")
        print(f"   - ❓ Fichiers non trouvés: {report['not_found']}")
        print(f"   - ❌ Échecs: {report['failed']}")
        
        broadcast_update()

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors de la ré-analyse: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': True, 'report': report})


# NOUVEAU: Fonctions pour SocketIO
@socketio.on('connect')
def handle_connect():
    print('🔌 Client connecté au WebSocket')

def broadcast_update():
    """Diffuse une mise à jour à tous les clients."""
    print("📡 Diffusion d'une mise à jour via WebSocket...")
    with app.app_context():
        socketio.emit('update_data', {'message': 'Les données ont été mises à jour !'})

@app.route('/api/delete_all_images', methods=['DELETE'])
def delete_all_images():
    """Supprime toutes les images et leurs données de la base de données et du dossier uploads."""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        # Récupérer tous les chemins de fichiers
        cursor.execute('SELECT id, file_path FROM images')
        images = cursor.fetchall()
        deleted_images = []
        for img in images:
            img_id, file_path = img
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_images.append({'id': img_id, 'file_path': file_path})
            except Exception as e:
                print(f"Erreur suppression fichier {file_path}: {e}")
        # Supprimer toutes les entrées de la table images
        cursor.execute('DELETE FROM images')
        deleted_data_count = cursor.rowcount
        conn.commit()
        conn.close()
        # Nettoyer le dossier uploads
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Erreur suppression fichier orphelin {file_path}: {e}")
        report = {
            'summary': {
                'total_images': len(images),
                'deleted_images_count': len(deleted_images),
                'deleted_data_count': deleted_data_count
            },
            'deleted_images': deleted_images,
            'deleted_data': [{'id': img[0]} for img in images]
        }
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        print(f"Erreur lors de la suppression de toutes les images: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/classifier_mode', methods=['GET'])
def get_classifier_mode():
    """Retourne le mode de classification actuel ('rules' ou 'ia')."""
    return jsonify({'mode': classifier.mode})

@app.route('/api/classifier_mode', methods=['POST'])
def set_classifier_mode():
    """Change le mode de classification ('rules' ou 'ia')."""
    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['rules', 'ia']:
        return jsonify({'success': False, 'error': 'Mode invalide'}), 400
    classifier.set_mode(mode)
    return jsonify({'success': True, 'mode': classifier.mode})

@app.route('/api/retrain_ia', methods=['POST'])
def retrain_ia():
    try:
        classifier.train_ia_model()
        return {'success': True, 'message': 'Modèle IA réentraîné. Voir la console pour les métriques.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    print("🚀 Initialisation de l'application...")
    
    # Initialiser et migrer la base de données
    init_db()
    migrate_database()
    
    # AJOUTEZ CES LIGNES MANQUANTES :
    print(f"\n🌐 Serveur démarré sur http://localhost:5000")
    print("📍 Routes disponibles:")
    print("   - / : Interface principale")
    print("   - /test : Test de connexion")
    print("   - /test-db : Test de la base de données")
    print("   - /upload : Upload d'images")
    
    # Lancer l'application avec SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)