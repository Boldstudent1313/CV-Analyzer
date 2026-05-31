
import os, re, fitz, torch, uuid, shutil
from flask import Flask, request, jsonify, render_template, session
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads_pro'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the high-fidelity semantic engine
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = " "
    for page in doc: text += page.get_text()
    doc.close()
    return re.sub(r'\\s+', ' ', text).strip()

def evaluate_alignment(cv_text, jd_text):
    cv_sentences = [s.strip() for s in re.split(r'[.!?\\n]', cv_text) if len(s.strip()) > 15]
    jd_sentences = [s.strip() for s in re.split(r'[.!?\\n]', jd_text) if len(s.strip()) > 10]
    if not cv_sentences or not jd_sentences: return 0.0, "Insufficient data"
    
    cv_emb = model.encode(cv_sentences, convert_to_tensor=True)
    jd_emb = model.encode(jd_sentences, convert_to_tensor=True)
    
    sim_matrix = util.cos_sim(jd_emb, cv_emb)
    max_sims = torch.max(sim_matrix, dim=1).values
    mean_score = torch.mean(max_sims).item()
    
    # Scale to 10 with intensity bias
    final_score = round(min(10.0, mean_score * 12), 1)
    return final_score, f"Semantic Match: {int(mean_score*100)}%"

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    jd = request.form.get('requirements', '')
    if file and file.filename.endswith('.pdf'):
        path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4()}.pdf')
        file.save(path)
        text = extract_text_from_pdf(path)
        os.remove(path)
        score, summary = evaluate_alignment(text, jd)
        return jsonify({'score': score, 'summary': summary})
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
