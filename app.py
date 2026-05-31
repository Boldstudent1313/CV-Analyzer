
import os, re, fitz, torch, uuid
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def rigorous_eval(cv_text, jd_text):
    if not cv_text or not jd_text: return 0.0, "Empty Data"
    cv_chunks = [s.strip() for s in re.split(r'[.!?\n]', cv_text) if len(s.strip()) > 20]
    jd_chunks = [s.strip() for s in re.split(r'[.!?\n]', jd_text) if len(s.strip()) > 15]
    
    cv_emb = model.encode(cv_chunks, convert_to_tensor=True)
    jd_emb = model.encode(jd_chunks, convert_to_tensor=True)
    
    sim_matrix = util.dot_score(jd_emb, cv_emb)
    max_sims = torch.max(sim_matrix, dim=1).values
    score = torch.mean(max_sims).item()
    
    final_score = round(min(10.0, max(0.0, score * 12)), 1)
    summary = f"Semantic Alignment: {int(score*100)}% | Model: MPNet-v1"
    return final_score, summary

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    jd = request.form.get('requirements', '')
    if file and file.filename.endswith('.pdf'):
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")
        file.save(path)
        
        doc = fitz.open(path)
        text = " ".join([p.get_text() for p in doc])
        doc.close()
        os.remove(path)
        
        score, summ = rigorous_eval(text, jd)
        return jsonify({'rating_out_of_10': score, 'summary': summ})
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
