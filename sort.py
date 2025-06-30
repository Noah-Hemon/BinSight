from flask import Flask, send_from_directory, request, jsonify, render_template_string
import os
import csv
from PIL import Image

app = Flask(__name__)
IMAGE_FOLDER = 'Data/test/'
CSV_FILE = 'Data/test_sorted.csv'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Créer les dossiers nécessaires
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs('Data', exist_ok=True)

# Helper pour lister les images
def get_image_files():
    if not os.path.exists(IMAGE_FOLDER):
        return []
    files = [f for f in os.listdir(IMAGE_FOLDER)
             if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
    files.sort()
    return files

# Helper pour extraire les infos d'une image
def get_image_info(filename):
    path = os.path.join(IMAGE_FOLDER, filename)
    try:
        with Image.open(path) as img:
            width, height = img.size
        filesize = os.path.getsize(path)
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'filesize': filesize
        }
    except Exception as e:
        print(f"Erreur lors de l'ouverture de {filename} : {e}")
        return {
            'filename': filename,
            'width': 0,
            'height': 0,
            'filesize': 0
        }

# Interface HTML pour le tri
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tri d'Images - Poubelles Pleines/Vides</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .btn {
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-full {
            background-color: #e53e3e;
            color: white;
        }
        .btn-full:hover {
            background-color: #c53030;
        }
        .btn-empty {
            background-color: #38a169;
            color: white;
        }
        .btn-empty:hover {
            background-color: #2f855a;
        }
        .btn-skip {
            background-color: #718096;
            color: white;
        }
        .btn-skip:hover {
            background-color: #4a5568;
        }
        .progress {
            background-color: #e2e8f0;
            border-radius: 10px;
            height: 20px;
            margin: 20px 0;
            overflow: hidden;
        }
        .progress-bar {
            background-color: #4299e1;
            height: 100%;
            transition: width 0.3s;
        }
        .info {
            background-color: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .no-images {
            text-align: center;
            color: #718096;
            font-size: 18px;
            margin: 50px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗑️ Tri d'Images - Poubelles</h1>
            <p>Classifiez chaque image comme "Pleine" ou "Vide"</p>
        </div>
        
        <div class="progress">
            <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
        </div>
        
        <div id="app">
            <div class="no-images">
                Chargement des images...
            </div>
        </div>
    </div>

    <script>
        let images = [];
        let currentIndex = 0;
        let labeled = new Set();

        // Charger les images disponibles
        function loadImages() {
            fetch('/api/images')
                .then(response => response.json())
                .then(data => {
                    images = data.images;
                    labeled = new Set(data.labeled);
                    console.log('Images chargées:', images.length);
                    console.log('Déjà étiquetées:', labeled.size);
                    displayCurrentImage();
                })
                .catch(error => {
                    console.error('Erreur:', error);
                    document.getElementById('app').innerHTML = '<div class="no-images">Erreur lors du chargement des images</div>';
                });
        }

        // Afficher l'image actuelle
        function displayCurrentImage() {
            const app = document.getElementById('app');
            
            if (images.length === 0) {
                app.innerHTML = '<div class="no-images">Aucune image trouvée dans le dossier Data/test/</div>';
                return;
            }

            // Trouver la prochaine image non étiquetée
            while (currentIndex < images.length && labeled.has(images[currentIndex].filename)) {
                currentIndex++;
            }

            if (currentIndex >= images.length) {
                app.innerHTML = `
                    <div class="no-images">
                        <h2>✅ Tri terminé !</h2>
                        <p>Toutes les images ont été classifiées.</p>
                        <p>Total traité: ${labeled.size} images</p>
                        <button class="btn btn-empty" onclick="downloadCSV()">Télécharger CSV</button>
                    </div>
                `;
                updateProgress();
                return;
            }

            const currentImage = images[currentIndex];
            
            app.innerHTML = `
                <div class="info">
                    <strong>Image:</strong> ${currentImage.filename}<br>
                    <strong>Dimensions:</strong> ${currentImage.width}x${currentImage.height}px<br>
                    <strong>Taille:</strong> ${(currentImage.filesize / 1024).toFixed(1)} Ko<br>
                    <strong>Progression:</strong> ${labeled.size}/${images.length} (${currentIndex + 1}/${images.length})
                </div>
                
                <div class="image-container">
                    <img src="/images/${currentImage.filename}" alt="${currentImage.filename}" />
                </div>
                
                <div class="buttons">
                    <button class="btn btn-full" onclick="labelImage('pleine')">
                        🗑️ Pleine
                    </button>
                    <button class="btn btn-empty" onclick="labelImage('vide')">
                        ✨ Vide
                    </button>
                    <button class="btn btn-skip" onclick="skipImage()">
                        ⏭️ Passer
                    </button>
                </div>
            `;
            
            updateProgress();
        }

        // Étiqueter une image
        function labelImage(label) {
            if (currentIndex >= images.length) return;
            
            const currentImage = images[currentIndex];
            
            fetch('/api/label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: currentImage.filename,
                    label: label,
                    width: currentImage.width,
                    height: currentImage.height,
                    filesize: currentImage.filesize
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    labeled.add(currentImage.filename);
                    currentIndex++;
                    displayCurrentImage();
                } else {
                    alert('Erreur lors de la sauvegarde: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Erreur:', error);
                alert('Erreur lors de la sauvegarde');
            });
        }

        // Passer une image
        function skipImage() {
            currentIndex++;
            displayCurrentImage();
        }

        // Mettre à jour la barre de progression
        function updateProgress() {
            const progress = images.length > 0 ? (labeled.size / images.length) * 100 : 0;
            document.getElementById('progress-bar').style.width = progress + '%';
        }

        // Télécharger le CSV
        function downloadCSV() {
            window.location.href = '/download-csv';
        }

        // Gestion des touches du clavier
        document.addEventListener('keydown', function(event) {
            if (event.key === '1' || event.key === 'f' || event.key === 'F') {
                labelImage('pleine');
            } else if (event.key === '2' || event.key === 'v' || event.key === 'V') {
                labelImage('vide');
            } else if (event.key === '3' || event.key === 's' || event.key === 'S') {
                skipImage();
            }
        });

        // Charger les images au démarrage
        loadImages();
    </script>
</body>
</html>
'''

# Route pour la page principale
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Route pour servir les images
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# Route pour obtenir la liste des images à trier
@app.route('/api/images')
def api_images():
    images = get_image_files()
    labeled = set()
    
    # Lire les images déjà étiquetées
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labeled.add(row['filename'])
        except Exception as e:
            print(f"Erreur lecture CSV: {e}")
    
    # Obtenir les infos pour toutes les images
    image_infos = []
    for img in images:
        info = get_image_info(img)
        image_infos.append(info)
    
    return jsonify({
        'images': image_infos,
        'labeled': list(labeled),
        'total': len(images),
        'remaining': len(images) - len(labeled)
    })

# Route pour enregistrer le label
@app.route('/api/label', methods=['POST'])
def api_label():
    try:
        data = request.get_json()
        filename = data.get('filename')
        label = data.get('label')
        width = data.get('width', 0)
        height = data.get('height', 0)
        filesize = data.get('filesize', 0)
        
        if not filename or not label:
            return jsonify({'success': False, 'message': 'Données manquantes'})
        
        if label not in ['pleine', 'vide']:
            return jsonify({'success': False, 'message': 'Label invalide'})
        
        # Créer le CSV s'il n'existe pas
        file_exists = os.path.exists(CSV_FILE)
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 'label', 'width', 'height', 'filesize', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'filename': filename,
                'label': label,
                'width': width,
                'height': height,
                'filesize': filesize,
                'timestamp': str(datetime.now())
            })
        
        print(f"✅ Image étiquetée: {filename} -> {label}")
        return jsonify({'success': True, 'message': f'Image étiquetée comme {label}'})
        
    except Exception as e:
        print(f"❌ Erreur étiquetage: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Route pour télécharger le CSV
@app.route('/download-csv')
def download_csv():
    if os.path.exists(CSV_FILE):
        return send_from_directory('.', CSV_FILE, as_attachment=True)
    else:
        return jsonify({'error': 'Fichier CSV non trouvé'}), 404

# Route de test
@app.route('/test')
def test():
    images = get_image_files()
    return jsonify({
        'status': 'OK',
        'images_found': len(images),
        'image_folder_exists': os.path.exists(IMAGE_FOLDER),
        'sample_images': images[:5]
    })

if __name__ == '__main__':
    # Importer datetime ici pour éviter les erreurs
    from datetime import datetime
    
    # Reset CSV at each run (optionnel)
    if os.path.exists(CSV_FILE):
        print(f"📄 CSV existant trouvé: {CSV_FILE}")
    else:
        print(f"📄 Nouveau CSV sera créé: {CSV_FILE}")
    
    print(f"📁 Dossier d'images: {IMAGE_FOLDER}")
    print(f"🖼️  Images trouvées: {len(get_image_files())}")
    print("🌐 Interface de tri disponible sur http://localhost:5001")
    
    app.run(debug=True, port=5001)