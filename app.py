
import os, re, fitz, torch, uuid, shutil
from flask import Flask, request, jsonify, render_template, session
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads_pro'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index(): 
    return render_template('index.html')

if __name__ == '__main__':
    # Render provides the PORT environment variable automatically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
