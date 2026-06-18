
from __future__ import annotations
from flask import Flask, render_template_string, request, jsonify
import os, re, json
from typing import List, Tuple

app = Flask(__name__)

# ---------------- Core text utils ----------------
_token_re = re.compile(r"[a-zA-Z0-9+#.]{2,}")
_stopwords = { 'and','or','the','for','with','to','of','in','on','a','an','is','are','as','by','be','at','from','this','that','you','your','we','our','they','their','them','it','its','if','else','then','will','shall','can','may','must','should' }


def normalize_text(s: str) -> str:
    s = '' if s is None else str(s)
    s = s.replace('\r',' ').replace('\t',' ')
    s = re.sub(r'\s+',' ', s).strip().lower()
    return s


def tokenize(s: str) -> List[str]:
    return _token_re.findall(normalize_text(s))

# ---------------- Keyword branch (fallback and for hybrid) ----------------

def extract_keywords(job_text: str) -> List[str]:
    toks = tokenize(job_text)
    out, seen = [], set()
    for t in toks:
        if t in _stopwords:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def keyword_coverage(cv_text: str, keywords: List[str]) -> Tuple[List[str], List[str]]:
    cv_norm = normalize_text(cv_text)
    matched = [k for k in keywords if k in cv_norm]
    missing = [k for k in keywords if k not in cv_norm]
    return matched, missing


def coverage_score(matched: List[str], total_keywords: int) -> int:
    if total_keywords <= 0:
        return 0
    return int(round((len(matched) / float(total_keywords)) * 100))

# ---------------- Semantic-lite branch (no training, no LLM) ----------------
# Use unsupervised TF-IDF and cosine to compute semantic-like similarity between
# job requirement units and CV chunks. This is not a keyword counter; cosine
# similarity over TF-IDF emphasizes informative n-grams and de-emphasizes stopwords.

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

_SPLIT_RE = re.compile(r"[\n\r]+|\.\s+|;\s+|\u2022\s+|\-\s+")


def split_units(text: str, cap: int = 1200) -> List[str]:
    parts = _SPILT_FIX if False else _SPLIT_RE.split(text or '')  # keep simple, avoid NameError
    out = [p.strip() for p in parts if len(p.strip()) > 6]
    return out[:cap]


def tfidf_semantic_score(job_text: str, cv_text: str):
    if TfidfVectorizer is None or cosine_similarity is None:
        return None
    j_units = split_units(job_text, cap=200)
    c_units = split_units(cv_text, cap=800)
    if not j_units or not c_units:
        return {'match_score': 0, 'matched': [], 'missing': j_units if j_units else [], 'pairs': []}
    # Build a joint TF-IDF over job+cv chunks to avoid OOV issues
    docs = j_units + c_units
    # Use char and word n-grams for robustness; build two vectorizers and concatenate similarities
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.99)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1)
    try:
        Xw = word_vec.fit_transform(docs)
        Xc = char_vec.fit_transform(docs)
    except Exception:
        return None
    Jw, Cw = Xw[: len(j_units), :], Xw[len(j_units) :, :]
    Jc, Cc = Xc[: len(j_units), :], Xc[len(j_units) :, :]
    # Cosine similarities
    Sw = cosine_similarity(Jw, Cw)
    Sc = cosine_similarity(Jc, Cc)
    # Combine
    S = 0.6 * Sw + 0.4 * Sc
    pairs = []
    covered, missing = [], []
    threshold = 0.35  # conservative default for unsupervised
    for i, ju in enumerate(j_units):
        jrow = S[i]
        jbest_idx = int(jrow.argmax())
        jbest = float(jrow[jbest_idx])
        cv_best = c_units[jbest_idx]
        pairs.append({'job_unit': ju, 'cv_unit': cv_best, 'score': jbest})
        (covered if jbest >= threshold else missing).append(ju)
    avg = sum(p['score'] for p in pairs) / max(1, len(pairs))
    match_score = int(round(max(0.0, min(1.0, avg)) * 100))
    return {'match_score': match_score, 'matched': covered, 'missing': missing, 'pairs': pairs}

# ---------------- Hybrid inference ----------------

def analyze_hybrid(cv_text: str, job_text: str):
    # Try semantic-lite first
    sem = tfidf_semantic_score(job_text, cv_text)
    if sem is None:
        # fall back to keyword coverage only
        kws = extract_keywords(job_text)
        matched, missing = keyword_coverage(cv_text, kws)
        score = coverage_score(matched, len(kws))
        return score, matched, missing
    # Blend with light keyword signal to stabilize edge cases
    kws = extract_keywords(job_text)
    kmatched, kmissing = keyword_coverage(cv_text, kws)
    kscore = coverage_score(kmatched, len(kws)) if kws else 0
    # Weighted blend: 80% semantic, 20% keyword
    score = int(round(0.8 * sem['match_score'] + 0.2 * kscore))
    # For UI compatibility, use matched/missing from semantic job units
    matched = sem['matched'][:50]
    missing = sem['missing'][:50]
    return score, matched, missing

# ---------------- PDF parsing ----------------

def parse_pdf_bytes(data: bytes) -> str:
    if not data:
        return ''
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ''
    tmp_path = '_upload_tmp.pdf'
    try:
        with open(tmp_path, 'wb') as tmp:
            tmp.write(data)
        text = extract_text(tmp_path) or ''
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text

# ---------------- UI template (unchanged layout) ----------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"> 
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
  <title>Bold CV Analyzer</title>
  <style>
    :root { --bg:#0b0f19; --panel:#151a29; --border:#22304a; --text:#e6ebff; --muted:#9fb0d7; --accent:#4da3ff; --good:#22c55e; --bad:#ef4444; }
    html,body { height:100%; }
    body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:var(--bg); color:var(--text); }
    header { padding:18px 22px; border-bottom:1px solid var(--border); background:linear-gradient(180deg, #0b0f19 0%, #0c1220 100%); position:sticky; top:0; z-index:10; }
    .brand { font-weight:800; letter-spacing:0.4px; }
    .accent { color:var(--accent); }
    .container { padding:22px; display:grid; grid-template-columns: 1fr 1fr; gap:22px; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
    h2 { margin:0 0 10px; font-size:18px; color:var(--text); }
    label { display:block; margin:8px 0 6px; color:var(--muted); font-size:13px; }
    input[type=file] { width:100%; padding:10px; background:#0f1524; color:var(--text); border:1px dashed var(--border); border-radius:10px; cursor:pointer; }
    textarea { width:100%; min-height:260px; padding:12px; background:#0f1524; color:var(--text); border:1px solid var(--border); border-radius:10px; resize:vertical; }
    .status { margin-top:8px; font-size:12px; color:var(--muted); }
    .actions { display:flex; gap:10px; margin-top:10px; }
    button { background:var(--accent); color:#051025; border:0; padding:10px 14px; border-radius:10px; font-weight:700; cursor:pointer; box-shadow: 0 6px 16px rgba(77,163,255,0.35); }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    pre { white-space:pre-wrap; background:#0f1524; padding:10px; border:1px solid var(--border); border-radius:10px; max-height:240px; overflow:auto; }
    .results { display:grid; grid-template-columns: 140px 1fr; gap:12px; align-items:start; }
    .score { font-size:52px; font-weight:900; line-height:1; }
    .score.good { color:var(--good); }
    .score.mid { color:#f59e0b; }
    .score.bad { color:#ef4444; }
    ul { margin:6px 0 0 16px; padding:0; }
    li { margin:4px 0; color:var(--muted); }
    @media (max-width: 900px){ .container{ grid-template-columns:1fr; } .results{ grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div class=\"brand\">Bold <span class=\"accent\">CV Analyzer</span></div>
  </header>
  <div class=\"container\">
    <section class=\"panel\" id=\"cv-panel\">
      <h2>CV (PDF)</h2>
      <label for=\"pdf\">Upload your CV</label>
      <input id=\"pdf\" name=\"file\" type=\"file\" accept=\".pdf\" />
      <div id=\"pdf-status\" class=\"status\"></div>
      <label for=\"pdf-text\">Extracted text (preview)</label>
      <pre id=\"pdf-text\"></pre>
    </section>

    <section class=\"panel\" id=\"job-panel\">
      <h2>Job Requirements</h2>
      <label for=\"job\">Paste the job description/requirements</label>
      <textarea id=\"job\" placeholder=\"Paste job requirements here...\"></textarea>
      <div class=\"actions\">
        <button id=\"analyze\">Analyze</button>
      </div>
      <div id=\"analyze-status\" class=\"status\"></div>
      <div id=\"results\" style=\"display:none; margin-top:10px;\">
        <div class=\"results\">
          <div class=\"score\" id=\"score\">0</div>
          <div>
            <div><strong>Matched keywords</strong></div>
            <ul id=\"matched\"></ul>
            <div style=\"margin-top:8px;\"><strong>Missing keywords</strong></div>
            <ul id=\"missing\"></ul>
            <div style=\"margin-top:8px;\"><strong>Recommendations</strong></div>
            <ul id=\"recs\"></ul>
          </div>
        </div>
      </div>
    </section>
  </div>

<script>
(function(){
  function el(id){ return document.getElementById(id); }
  function statusPDF(t){ el('pdf-status').textContent = t; }
  function statusAnalyze(t){ el('analyze-status').textContent = t; }
  function setScore(val){
    const s = el('score'); s.textContent = val; s.classList.remove('good','mid','bad');
    if(val >= 75) s.classList.add('good'); else if(val >= 45) s.classList.add('mid'); else s.classList.add('bad');
  }
  function bindPDF(){
    const input = el('pdf') || document.querySelector('input[type=\\"file\\"][name=\\"file\\"]');
    if(!input){ statusPDF('File input not found'); return; }
    input.accept = '.pdf';
    input.addEventListener('change', async function(){
      if(!input.files || !input.files[0]){ statusPDF('No file selected'); return; }
      const f = input.files[0]; if(!/\\\\.pdf$/i.test(f.name)){ statusPDF('Please select a PDF'); return; }
      statusPDF('Parsing PDF...');
      try{ const fd = new FormData(); fd.append('file', f); const resp = await fetch('/api/parse_pdf', {method:'POST', body:fd});
        const data = await resp.json(); if(!data.ok){ statusPDF('Error: ' + (data.error || 'Unknown error')); return; }
        statusPDF('Parsed ' + (data.chars||0) + ' characters'); el('pdf-text').textContent = data.text || ''; }
      catch(e){ statusPDF('Upload failed: ' + e.message); }
    }); }
  async function analyze(){
    const cv = el('pdf-text').textContent || ''; const job = el('job').value || '';
    if(!cv.trim()){ statusAnalyze('Please upload and parse a CV first.'); return; }
    if(!job.trim()){ statusAnalyze('Please paste the job requirements.'); return; }
    statusAnalyze('Analyzing...'); el('analyze').disabled = true;
    try{ const resp = await fetch('/api/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ cv_text: cv, job_text: job }) });
      const data = await resp.json(); if(!data.ok){ statusAnalyze('Error: ' + (data.error || 'Unknown error')); return; }
      statusAnalyze(''); setScore(data.match_score || 0);
      const matched = el('matched'); matched.innerHTML=''; const missing = el('missing'); missing.innerHTML=''; const recs = el('recs'); recs.innerHTML='';
      (data.matched_keywords||[]).slice(0,50).forEach(k=>{ const li=document.createElement('li'); li.textContent = k; matched.appendChild(li); });
      (data.missing_keywords||[]).slice(0,50).forEach(k=>{ const li=document.createElement('li'); li.textContent = k; missing.appendChild(li); });
      (data.recommendations||[]).slice(0,10).forEach(r=>{ const li=document.createElement('li'); li.textContent = r; recs.appendChild(li); });
      el('results').style.display = ''; }
    catch(e){ statusAnalyze('Analyze failed: ' + e.message); }
    finally { el('analyze').disabled = false; } }
  function bindAnalyze(){ el('analyze').addEventListener('click', analyze); }
  if(document.readyState === 'loading'){ document.addEventListener('DOMContentLoaded', ()=>{ bindPDF(); bindAnalyze(); }); } else { bindPDF(); bindAnalyze(); }
})();
</script>
</body>
</html>
"""

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/health', methods=['GET'])
def health():
    return ('ok', 200)

@app.route('/api/parse_pdf', methods=['POST'])
def parse_pdf():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file part'}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'ok': False, 'error': 'No selected file'}), 400
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'ok': False, 'error': 'Only PDF files are supported'}), 400
    try:
        data = f.read()
        text = parse_pdf_bytes(data)
        preview = (text or '')[:20000]
        return jsonify({'ok': True, 'text': preview, 'chars': len(text or '')})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    try:
        data = request.get_json(force=True, silent=False) or {}
        cv = data.get('cv_text') or ''
        job = data.get('job_text') or ''
        if not cv.strip() or not job.strip():
            return jsonify({'ok': False, 'error': 'cv_text and job_text are required'}), 400
        score, matched, missing = analyze_hybrid(cv, job)
        # Simple recommendations
        recs = []
        if missing:
            recs.append('Consider adding evidence for: ' + ', '.join(missing[:8]) + '.')
        if score < 60:
            recs.append('Tailor your CV summary to mirror key role requirements and metrics.')
        return jsonify({'ok': True, 'match_score': score, 'matched_keywords': matched, 'missing_keywords': missing, 'recommendations': recs})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
