from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import sqlite3
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import traceback
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'binsight-secret-key-2023'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

def migrate_database():
    """Migrer la base de données pour ajouter les colonnes manquantes"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    try:
        # Vérifier la structure actuelle de la table
        cursor.execute("PRAGMA table_info(images)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Colonnes existantes: {columns}")
        
        # Ajouter user_annotation si elle n'existe pas
        if 'user_annotation' not in columns:
            print("Ajout de la colonne user_annotation...")
            cursor.execute('ALTER TABLE images ADD COLUMN user_annotation TEXT')
            conn.commit()
            print("✓ Colonne user_annotation ajoutée")
        else:
            print("✓ Colonne user_annotation déjà présente")
            
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
            confidence REAL DEFAULT 0.0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Base de données initialisée avec la structure complète")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class FeatureExtractor:
    def extract_features(self, image_path):
        """Extraire les caractéristiques d'une image"""
        try:
            print(f"Extraction des caractéristiques pour: {image_path}")
            
            # Vérifier que le fichier existe
            if not os.path.exists(image_path):
                raise Exception(f"Fichier non trouvé: {image_path}")
            
            # Taille du fichier
            file_size = os.path.getsize(image_path)
            print(f"Taille du fichier: {file_size} bytes")
            
            # Ouvrir avec PIL
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
            print(f"Luminosité: {brightness:.1f}")
            
            # Valeurs par défaut pour OpenCV (au cas où ça échoue)
            contrast = 30.0
            edge_density = 0.1
            
            try:
                # Essayer OpenCV
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    contrast = gray.std()
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                    print(f"OpenCV - Contraste: {contrast:.1f}, Densité contours: {edge_density:.3f}")
                else:
                    print("OpenCV n'a pas pu lire l'image, utilisation des valeurs par défaut")
            except Exception as cv_error:
                print(f"Erreur OpenCV (ignorée): {cv_error}")
            
            features = {
                'file_size': file_size,
                'width': width,
                'height': height,
                'avg_red': float(avg_red),
                'avg_green': float(avg_green),
                'avg_blue': float(avg_blue),
                'brightness': float(brightness),
                'contrast_level': float(contrast),
                'edge_density': float(edge_density)
            }
            
            print(f"✓ Caractéristiques extraites avec succès")
            return features
            
        except Exception as e:
            print(f"❌ Erreur extraction: {e}")
            print(traceback.format_exc())
            return None

class SimpleClassifier:
    def __init__(self):
        self.rules = {
            'dark_threshold': 80,
            'bright_threshold': 150,
            'size_threshold': 500000,
            'contrast_threshold': 30,
            'high_contrast_threshold': 60,
            'edge_threshold': 0.1,
            'high_edge_threshold': 0.2
        }
    
    def classify(self, features):
        """Classifier l'image selon des règles simples"""
        if not features:
            return 'inconnu', 0.0
        
        brightness = features['brightness']
        file_size = features['file_size']
        contrast = features['contrast_level']
        edge_density = features['edge_density']
        
        print(f"Classification - Luminosité: {brightness:.1f}, Contraste: {contrast:.1f}, Contours: {edge_density:.3f}")
        
        # Règles de classification
        if brightness < self.rules['dark_threshold']:
            print("→ Règle: Image sombre = pleine")
            return 'pleine', 0.75
        elif brightness > self.rules['bright_threshold']:
            print("→ Règle: Image claire = vide")
            return 'vide', 0.80
        elif edge_density > self.rules['high_edge_threshold']:
            print("→ Règle: Beaucoup de contours = pleine")
            return 'pleine', 0.85
        elif edge_density < self.rules['edge_threshold'] and contrast < self.rules['contrast_threshold']:
            print("→ Règle: Peu de contours + faible contraste = vide")
            return 'vide', 0.70
        elif contrast > self.rules['high_contrast_threshold']:
            print("→ Règle: Contraste élevé = pleine")
            return 'pleine', 0.75
        elif file_size > self.rules['size_threshold'] and edge_density > 0.05:
            print("→ Règle: Gros fichier + contours moyens = pleine")
            return 'pleine', 0.65
        else:
            print("→ Règle: Par défaut = vide")
            return 'vide', 0.55

# Initialiser les composants
feature_extractor = FeatureExtractor()
classifier = SimpleClassifier()

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
            features = feature_extractor.extract_features(file_path)
            
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
                        comment, auto_classification, user_annotation, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename, file_path, features['file_size'], features['width'], 
                    features['height'], features['avg_red'], features['avg_green'], 
                    features['avg_blue'], features['contrast_level'], features['brightness'],
                    features['edge_density'], location if location else None, 
                    bin_type if bin_type else None, comment if comment else None, 
                    auto_classification, user_annotation if user_annotation else None, confidence
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
            extension = file.filename.split('.')[-1] if '.' in file.filename else 'aucune'
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
def api_recent_reports():
    """API pour les derniers signalements"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''SELECT id, filename, location, manual_annotation, auto_classification, 
            upload_date, bin_type, file_path, user_annotation FROM images 
            ORDER BY upload_date DESC LIMIT 10''')
        
        images = cursor.fetchall()
        conn.close()
        
        reports = []
        for img in images:
            status = img[8] or img[3] or img[4] or 'Non classée'
            
            try:
                upload_time = datetime.strptime(img[5], '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - upload_time
                if time_diff.days > 0:
                    time_str = f"{time_diff.days} j"
                elif time_diff.seconds >= 3600:
                    time_str = f"{time_diff.seconds // 3600} h"
                else:
                    time_str = f"{time_diff.seconds // 60} min"
            except:
                time_str = "Récent"
            
            reports.append({
                'id': img[0],
                'filename': img[1],
                'location': img[2] or 'Non spécifiée',
                'status': status,
                'date': time_str,
                'bin_type': img[6] or 'Standard',
                'image_url': url_for('uploaded_file', filename=img[1]) if img[1] else None
            })
        
        return jsonify(reports)
        
    except Exception as e:
        print(f"Erreur API recent_reports: {e}")
        return jsonify([])

@app.route('/api/poubelles')
def api_poubelles():
    """API pour les données des poubelles sur la carte avec vraies coordonnées depuis la DB"""
    try:
        conn = sqlite3.connect('binsight.db')
        cursor = conn.cursor()
        
        cursor.execute('''SELECT 
            id, filename, location, manual_annotation, auto_classification, 
            upload_date, confidence, user_annotation, bin_type, comment 
            FROM images 
            ORDER BY upload_date DESC LIMIT 50''')
        
        images = cursor.fetchall()
        conn.close()
        
        poubelles = []
        for i, img in enumerate(images):
            # Déterminer le statut final (priorité: user_annotation > manual_annotation > auto_classification)
            status = img[7] or img[3] or img[4] or 'non classée'
            
            # Calculer le temps écoulé depuis l'upload
            try:
                upload_time = datetime.strptime(img[5], '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - upload_time
                if time_diff.days > 0:
                    time_str = f"{time_diff.days} j"
                elif time_diff.seconds >= 3600:
                    time_str = f"{time_diff.seconds // 3600} h"
                else:
                    time_str = f"{time_diff.seconds // 60} min"
            except:
                time_str = "Récent"
            
            # 🎯 EXTRACTION DES COORDONNÉES DEPUIS LA DB
            lat, lng = extract_coordinates_from_location(img[2])  # img[2] = location
            
            poubelles.append({
                'id': img[0],
                'lat': lat,
                'lng': lng,
                'status': status,
                'time': time_str,
                'location': img[2] or f'Point {i+1}',
                'confidence': img[6] or 0.0,
                'bin_type': img[8] or 'Standard',
                'comment': img[9]
            })
        
        print(f"✅ {len(poubelles)} poubelles chargées avec coordonnées depuis la DB")
        return jsonify(poubelles)
        
    except Exception as e:
        print(f"❌ Erreur API poubelles: {e}")
        return jsonify([])

def extract_coordinates_from_location(location_str):
    """Extraire les coordonnées lat/lng depuis le string de localisation"""
    try:
        if not location_str:
            # Coordonnées par défaut (Paris)
            return 48.8566, 2.3522
        
        # Si la location contient un tuple, l'extraire
        if '(' in location_str and ')' in location_str and ',' in location_str:
            # Format: "(-30.250704, -57.148525)" ou "Location Name (-30.250704, -57.148525)"
            
            # Trouver le contenu entre parenthèses
            start = location_str.rfind('(')
            end = location_str.rfind(')')
            
            if start != -1 and end != -1 and end > start:
                coord_str = location_str[start+1:end]
                
                # Séparer par la virgule
                parts = [part.strip() for part in coord_str.split(',')]
                
                if len(parts) == 2:
                    try:
                        lat = float(parts[0])
                        lng = float(parts[1])
                        
                        # Vérifier que les coordonnées sont dans des plages valides
                        if -90 <= lat <= 90 and -180 <= lng <= 180:
                            print(f"✅ Coordonnées extraites: ({lat}, {lng}) depuis '{location_str}'")
                            return lat, lng
                        else:
                            print(f"⚠️ Coordonnées hors limites: ({lat}, {lng})")
                    except ValueError as e:
                        print(f"⚠️ Erreur conversion coordonnées: {e}")
        
        # Si pas de coordonnées trouvées, essayer de deviner selon le nom de lieu
        return get_default_coordinates_by_name(location_str)
        
    except Exception as e:
        print(f"❌ Erreur extraction coordonnées: {e}")
        return 48.8566, 2.3522  # Paris par défaut

def get_default_coordinates_by_name(location_name):
    """Obtenir des coordonnées par défaut selon le nom de lieu"""
    
    if not location_name:
        return 48.8566, 2.3522  # Paris
    
    location_lower = location_name.lower()
    
    # Dictionnaire des villes avec coordonnées
    city_coordinates = {
        'paris': (48.8566, 2.3522),
        'lyon': (45.7640, 4.8357),
        'marseille': (43.2965, 5.3698),
        'lille': (50.6292, 3.0573),
        'nantes': (47.2184, -1.5536),
        'bordeaux': (44.8378, -0.5792),
        'toulouse': (43.6047, 1.4442),
        'nice': (43.7102, 7.2620),
        'strasbourg': (48.5734, 7.7521),
        'rennes': (48.1173, -1.6778),
        'reims': (49.2583, 4.0317),
        'angers': (47.4784, -0.5632),
        'le havre': (49.4944, 0.1079),
        'montpellier': (43.6119, 3.8772),
        'nancy': (48.6921, 6.1844)
    }
    
    # Rechercher une correspondance
    for city, coords in city_coordinates.items():
        if city in location_lower:
            # Ajouter une petite variation pour éviter la superposition
            lat_offset = (hash(location_name) % 100 - 50) * 0.001  # ±0.05 degrés max
            lng_offset = (hash(location_name) % 100 - 50) * 0.001
            
            final_lat = coords[0] + lat_offset
            final_lng = coords[1] + lng_offset
            
            print(f"📍 Coordonnées par nom '{location_name}' -> {city}: ({final_lat:.4f}, {final_lng:.4f})")
            return final_lat, final_lng
    
    # Si aucune ville reconnue, utiliser des coordonnées aléatoires en France
    print(f"🎲 Coordonnées aléatoires pour '{location_name}'")
    
    # Zone France métropolitaine approximative
    france_bounds = {
        'lat_min': 41.3,
        'lat_max': 51.1,
        'lng_min': -5.1,
        'lng_max': 9.6
    }
    
    # Générer des coordonnées pseudo-aléatoires basées sur le hash du nom
    hash_val = hash(location_name)
    lat = france_bounds['lat_min'] + (hash_val % 1000) / 1000 * (france_bounds['lat_max'] - france_bounds['lat_min'])
    lng = france_bounds['lng_min'] + ((hash_val // 1000) % 1000) / 1000 * (france_bounds['lng_max'] - france_bounds['lng_min'])
    
    return round(lat, 6), round(lng, 6)

# 🔧 FONCTION DE TEST pour vérifier l'extraction
@app.route('/test-coordinates')
def test_coordinates():
    """Route de test pour vérifier l'extraction de coordonnées"""
    test_locations = [
        "(-30.250704671553954, -57.14852536982011)",
        "Paris (-32.89257311452824, -53.43179943666392)",
        "Lyon (45.7640, 4.8357)",
        "Rue de la Paix, Paris",
        "Marseille",
        "Localisation inconnue",
        "",
        None
    ]
    
    results = []
    for loc in test_locations:
        lat, lng = extract_coordinates_from_location(loc)
        results.append({
            'input': loc,
            'output': f"({lat}, {lng})",
            'lat': lat,
            'lng': lng
        })
    
    return jsonify({
        'test_results': results,
        'status': 'Test coordinates extraction completed'
    })

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

if __name__ == '__main__':
    print("🚀 Initialisation de l'application...")
    
    # Initialiser et migrer la base de données
    init_db()
    migrate_database()
    
    # Test de la structure
    print("\n📊 Test de la structure de la base de données...")
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) FROM images")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"✅ Colonnes dans la table: {len(columns)}")
    print(f"✅ user_annotation présente: {'user_annotation' in columns}")
    print(f"✅ Images en base: {count}")
    
    print(f"\n🌐 Serveur démarré sur http://localhost:5000")
    print("📍 Routes disponibles:")
    print("   - / : Interface principale")
    print("   - /test : Test de connexion")
    print("   - /test-db : Test de la base de données")
    print("   - /upload : Upload d'images")
    
    app.run(debug=True, host='0.0.0.0', port=5000)