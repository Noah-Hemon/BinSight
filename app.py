from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import sqlite3
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import base64
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'binsight-secret-key-2023'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

def init_db():
    """Initialiser la base de données avec le champ user_annotation"""
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class FeatureExtractor:
    def extract_features(self, image_path):
        """Extraire les caractéristiques d'une image"""
        try:
            # Taille du fichier
            file_size = os.path.getsize(image_path)
            
            # Ouvrir avec PIL
            pil_image = Image.open(image_path)
            width, height = pil_image.size
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Couleurs moyennes
            np_image = np.array(pil_image)
            avg_red = np.mean(np_image[:, :, 0])
            avg_green = np.mean(np_image[:, :, 1])
            avg_blue = np.mean(np_image[:, :, 2])
            
            # Luminosité
            brightness = np.mean(np_image)
            
            # Ouvrir avec OpenCV pour les caractéristiques avancées
            cv_image = cv2.imread(image_path)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Contraste
            contrast = gray.std()
            
            # Détection de contours
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
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
        except Exception as e:
            print(f"Erreur extraction: {e}")
            return None

class SimpleClassifier:
    def __init__(self):
        # Règles de classification
        self.rules = {
            'dark_threshold': 80,
            'size_threshold': 500000,  # 500KB
            'contrast_threshold': 30,
            'edge_threshold': 0.1
        }
    
    def classify(self, features):
        """Classifier l'image selon des règles simples"""
        if not features:
            return 'inconnu', 0.0
        
        brightness = features['brightness']
        file_size = features['file_size']
        contrast = features['contrast_level']
        edge_density = features['edge_density']
        
        confidence = 0.0
        
        # Règle 1: Image sombre + gros fichier = poubelle pleine
        if brightness < self.rules['dark_threshold'] and file_size > self.rules['size_threshold']:
            confidence = 0.85
            return 'pleine', confidence
        
        # Règle 2: Faible contraste + peu de contours = poubelle vide
        if contrast < self.rules['contrast_threshold'] and edge_density < self.rules['edge_threshold']:
            confidence = 0.75
            return 'vide', confidence
        
        # Règle 3: Luminosité élevée = poubelle vide
        if brightness > 150:
            confidence = 0.80
            return 'vide', confidence
        
        # Règle 4: Densité de contours élevée = poubelle pleine
        if edge_density > 0.2:
            confidence = 0.88
            return 'pleine', confidence
        
        # Par défaut
        confidence = 0.60
        return 'vide', confidence

# Initialiser les composants
feature_extractor = FeatureExtractor()
classifier = SimpleClassifier()

@app.route('/')
def index():
    return render_template('BinSight.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gérer l'upload d'images avec extraction de caractéristiques et annotation utilisateur"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Aucun fichier trouvé'})
        
        file = request.files['file']
        location = request.form.get('location', '')
        bin_type = request.form.get('bin_type', '')
        comment = request.form.get('comment', '')
        user_annotation = request.form.get('user_annotation', '')  # Nouvelle annotation utilisateur
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'})
        
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extraire les caractéristiques
            features = feature_extractor.extract_features(file_path)
            
            if features:
                # Classification automatique (seulement si l'utilisateur n'a pas choisi)
                if user_annotation:
                    auto_classification = user_annotation
                    confidence = 1.0  # Confiance maximale pour l'annotation utilisateur
                    message = f'Annotation utilisateur: Poubelle {user_annotation}'
                else:
                    auto_classification, confidence = classifier.classify(features)
                    message = f'Analyse automatique: Poubelle {auto_classification} détectée'
                
                # Sauvegarder en base de données
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
                    features['edge_density'], location, bin_type, comment, 
                    auto_classification, user_annotation, confidence
                ))
                
                image_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                return jsonify({
                    'success': True,
                    'message': message,
                    'classification': auto_classification,
                    'confidence': f'{confidence*100:.1f}%',
                    'features': features,
                    'image_id': image_id,
                    'image_url': url_for('uploaded_file', filename=filename),
                    'is_user_annotation': bool(user_annotation)
                })
            else:
                return jsonify({'success': False, 'message': 'Erreur lors de l\'analyse de l\'image'})
        else:
            return jsonify({'success': False, 'message': 'Format de fichier non supporté. Utilisez JPG, PNG ou JPEG.'})
    
    except Exception as e:
        print(f"Erreur upload: {e}")
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
        return jsonify({'success': True, 'message': f'Image annotée comme {annotation}'})
    return jsonify({'success': False, 'message': 'Annotation invalide'})

@app.route('/api/stats')
def api_stats():
    """API pour les statistiques du tableau de bord"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    # Total d'images aujourd'hui
    cursor.execute('''
        SELECT COUNT(*) FROM images 
        WHERE date(upload_date) = date('now')
    ''')
    today_uploads = cursor.fetchone()[0]
    
    # Poubelles pleines (user_annotation, manual_annotation ou auto)
    cursor.execute('''
        SELECT COUNT(*) FROM images 
        WHERE user_annotation = 'pleine' OR 
              manual_annotation = 'pleine' OR 
              (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'pleine')
    ''')
    full_bins = cursor.fetchone()[0]
    
    # Précision (comparer auto avec manual seulement)
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN manual_annotation = auto_classification THEN 1 ELSE 0 END) as correct
        FROM images 
        WHERE manual_annotation IS NOT NULL AND user_annotation IS NULL
    ''')
    precision_data = cursor.fetchone()
    accuracy = 0.0
    if precision_data[0] > 0:
        accuracy = (precision_data[1] / precision_data[0]) * 100
    
    # Données mensuelles pour les graphiques
    cursor.execute('''
        SELECT 
            strftime('%m', upload_date) as month,
            COUNT(*) as total,
            SUM(CASE WHEN 
                user_annotation = 'vide' OR 
                (user_annotation IS NULL AND manual_annotation = 'vide') OR 
                (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'vide') 
                THEN 1 ELSE 0 END) as empty,
            SUM(CASE WHEN 
                user_annotation = 'pleine' OR 
                (user_annotation IS NULL AND manual_annotation = 'pleine') OR 
                (user_annotation IS NULL AND manual_annotation IS NULL AND auto_classification = 'pleine') 
                THEN 1 ELSE 0 END) as full
        FROM images 
        WHERE strftime('%Y', upload_date) = strftime('%Y', 'now')
        GROUP BY strftime('%m', upload_date)
        ORDER BY month
    ''')
    monthly_data = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'today_uploads': today_uploads,
        'full_bins': full_bins,
        'accuracy': round(accuracy, 1),
        'monthly_data': monthly_data
    })

@app.route('/api/recent_reports')
def api_recent_reports():
    """API pour les derniers signalements"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, location, manual_annotation, auto_classification, 
               upload_date, bin_type, file_path, user_annotation 
        FROM images 
        ORDER BY upload_date DESC 
        LIMIT 10
    ''')
    
    images = cursor.fetchall()
    conn.close()
    
    reports = []
    for img in images:
        # Déterminer le statut final (priorité : user_annotation > manual_annotation > auto_classification)
        status = img[8] or img[3] or img[4]  # user_annotation, manual_annotation ou auto_classification
        status = status if status else 'Non classée'
        
        # Calculer le temps écoulé
        upload_time = datetime.strptime(img[5], '%Y-%m-%d %H:%M:%S')
        time_diff = datetime.now() - upload_time
        if time_diff.seconds < 3600:
            time_str = f"{time_diff.seconds // 60} min"
        else:
            time_str = f"{time_diff.seconds // 3600} h"
        
        reports.append({
            'id': img[0],
            'filename': img[1],
            'location': img[2] or 'Non spécifiée',
            'status': status,
            'date': time_str,
            'bin_type': img[6] or 'Standard',
            'image_url': url_for('uploaded_file', filename=img[1]) if img[7] else None
        })
    
    return jsonify(reports)

@app.route('/api/poubelles')
def api_poubelles():
    """API pour les données des poubelles sur la carte"""
    conn = sqlite3.connect('binsight.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, location, manual_annotation, auto_classification, 
               upload_date, confidence, user_annotation 
        FROM images 
        ORDER BY upload_date DESC 
        LIMIT 20
    ''')
    
    images = cursor.fetchall()
    conn.close()
    
    # Coordonnées factices pour Paris (vous pourriez stocker de vraies coordonnées)
    base_coords = [
        (48.8566, 2.3522), (48.8575, 2.3585), (48.8550, 2.3465),
        (48.8585, 2.3376), (48.8595, 2.3476), (48.8545, 2.3376),
        (48.8535, 2.3476), (48.8525, 2.3576), (48.8515, 2.3676),
        (48.8505, 2.3776), (48.8495, 2.3876), (48.8485, 2.3976),
        (48.8475, 2.4076), (48.8465, 2.4176), (48.8455, 2.4276),
        (48.8445, 2.4376), (48.8435, 2.4476), (48.8425, 2.4576),
        (48.8415, 2.4676), (48.8405, 2.4776)
    ]
    
    poubelles = []
    for i, img in enumerate(images):
        if i < len(base_coords):
            # Déterminer le statut (priorité : user_annotation > manual > auto)
            status = img[7] or img[3] or img[4]  # user_annotation, manual ou auto
            status = status if status else 'vide'
            
            # Calculer le temps écoulé
            upload_time = datetime.strptime(img[5], '%Y-%m-%d %H:%M:%S')
            time_diff = datetime.now() - upload_time
            if time_diff.seconds < 3600:
                time_str = f"{time_diff.seconds // 60} min"
            else:
                time_str = f"{time_diff.seconds // 3600} h"
            
            poubelles.append({
                'id': img[0],
                'lat': base_coords[i][0],
                'lng': base_coords[i][1],
                'status': status,
                'time': time_str,
                'location': img[2] or f'Location {i+1}',
                'confidence': img[6] or 0.0
            })
    
    return jsonify(poubelles)

@app.route('/api/image/<int:image_id>')
def get_image_details(image_id):
    """Récupérer les détails d'une image"""
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

if __name__ == '__main__':
    init_db()
    app.run(debug=False)  # Désactiver le debug pour éviter le problème watchdog