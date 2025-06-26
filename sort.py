from flask import Flask, send_from_directory, request, jsonify, render_template_string
import os
import csv
from PIL import Image

app = Flask(__name__)
IMAGE_FOLDER = 'Data/test/'
CSV_FILE = 'Data/test_sorted.csv'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Helper pour lister les images
def get_image_files():
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
            img_small = img.resize((50, 50))
            pixels = list(img_small.getdata())
            avg_color = tuple(sum(c) // len(c) for c in zip(*pixels))
            dominant_color = '#%02x%02x%02x' % avg_color
        filesize = os.path.getsize(path)
        return width, height, filesize, dominant_color
    except Exception as e:
        print(f"Erreur lors de l'ouverture de {filename} : {e}")
        return None, None, None, None

# Route pour la page principale
@app.route('/')
def index():
    with open('sort.html', encoding='utf-8') as f:
        return f.read()

# Route pour servir les images
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# Route pour obtenir la liste des images à trier
@app.route('/api/images')
def api_images():
    images = get_image_files()
    labeled = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            labeled = set(row['filename'] for row in reader if row and 'filename' in row and row['filename'])
    # On vérifie que l'image est bien lisible
    to_label = []
    for img in images:
        if img not in labeled:
            width, height, filesize, dominant_color = get_image_info(img)
            if width is not None:
                to_label.append(img)
    return jsonify(to_label)

# Route pour enregistrer le label
@app.route('/api/label', methods=['POST'])
def api_label():
    data = request.json
    filename = data['filename']
    label = data['label']
    # Remplacer 'full' par 'overflowing' si besoin
    if label == 'full':
        label = 'overflowing'
    width, height, filesize, dominant_color = get_image_info(filename)
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'width', 'height', 'filesize', 'dominant_color', 'label'])
        writer.writerow([filename, width, height, filesize, dominant_color, label])
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
