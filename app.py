from __future__ import annotations

import os
import re
import io
import json
import time
from typing import List, Dict, Any, Tuple

from flask import Flask, render_template_string, request, jsonify

# Optional imports (graceful degradation)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

# Optional web enrichment library
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

app = Flask(__name__)

# ---------------- Precompiled regexes and constants ----------------
_space_re = re.compile(r'\s+')
_token_re = re.compile(r"[a-zA-Z0-9+#.]{2,}")
_SPLIT_RE = re.compile(r"[\n\r]+|\.?\s*[.;]\s+|\u2022\s+|-{1}\s+")
_NP_RE = re.compile(r'(?:[a-z0-9+#.]+(?:\s+|\-)){0,3}(?:[a-z0-9+#.]+)')

_stopwords = {
    'and','or','the','for','with','to','of','in','on','a','an','is','are','as','by','be','at','from',
    'this','that','you','your','we','our','they','their','them','it','its','if','else','then','will',
    'shall','can','may','must','should'
}

# Aliases for common requirement phrases to improve recall
ALIASES: Dict[str, List[str]] = {
    'society involvement': [
        'student society', 'student societies', 'societies', 'society',
        'club', 'clubs', 'association', 'associations', 'chapter',
        'student chapter', 'community work', 'community service',
        'volunteering', 'volunteer work', 'extracurricular', 'campus activities'
    ],
    'leadership': [
        'led', 'team lead', 'president', 'captain', 'chair', 'head',
        'organized', 'coordinated', 'mentored', 'supervised'
    ],
    'communication skills': [
        'presentation', 'presented', 'public speaking', 'stakeholder communication',
        'report', 'reporting', 'documentation', 'writing', 'client communication'
    ],
    'teamwork': [
        'collaborated', 'cross-functional', 'pair programming', 'scrum',
        'agile', 'team project', 'coordination'
    ],
    'open source': [
        'open-source', 'github contributions', 'maintainer', 'pull request',
        'issue triage', 'community contributions'
    ]
}

# Small background corpus to “pre-train” TF-IDF for better general semantic space
BACKGROUND_CORPUS = [
    # General skills/phrases
    "leadership team coordination stakeholder communication project planning mentoring coaching",
    "problem solving analytical thinking initiative ownership adaptability collaboration",
    "writing documentation reporting presentations public speaking",
    "applied statistics machine learning data pipelines visualization dashboards",
    "rest apis microservices databases caching observability testing deployment ci cd",
    "product strategy user research roadmap experimentation metrics kpi alignment",
    "community involvement student societies volunteering clubs associations chapters"
]


# ---------------- Text utilities ----------------
def normalize_text(s: str) -> str:
    s = '' if s is None else str(s)
    s = s.replace('\r',' ').replace('\t',' ')
    s = _space_re.sub(' ', s).strip().lower()
    return s

def tokenize(s: str) -> List[str]:
    return _token_re.findall(normalize_text(s))

def noun_phrases(text: str) -> List[str]:
    tx = normalize_text(text)
    cands = _NP_RE.findall(tx)
    return [c.strip() for c in cands if len(c.strip()) >= 6][:200]

def augment_with_np(units: List[str]) -> List[str]:
    out = []
    for u in units:
        nps = noun_phrases(u)[:5]
        if nps:
            out.append(u + ' ' + ' '.join(nps))
        else:
            out.append(u)
    return out

def _split_safe(text: str) -> List[str]:
    return _SPLIT_RE.split(text or '')

def split_units(text: str, cap: int = 1200) -> List[str]:
    cap = 1200 if cap is None or cap > 1200 else cap
    parts = _split_safe(text)
    out = [p.strip() for p in parts if len(p.strip()) > 6]
    return out[:cap]

def expand_unit_aliases(u: str) -> List[str]:
    base = normalize_text(u)
    alts = set([base])
    for k, vals in ALIASES.items():
        if k in base:
            for v in vals:
                alts.add(normalize_text(v))
    return list(alts)

def keyword_present(cv_norm: str, unit_norm: str) -> bool:
    if unit_norm in cv_norm:
        return True
    utoks = set(tokenize(unit_norm)) - _stopwords
    if not utoks:
        return False
    hits = sum(1 for t in utoks if t in cv_norm)
    return hits >= max(1, len(utoks) // 2)


# ---------------- Optional web enrichment (duckduckgo_search) ----------------
def web_enrich_terms(terms: List[str], max_terms: int = 3, per_term: int = 2, timeout: float = None) -> List[str]:
    if os.getenv('GOOGLE_WEB_ENABLED', '0') != '1':
        return []
    if DDGS is None:
        return []
    timeout = timeout or float(os.getenv('WEB_ENRICH_TIMEOUT', '1.2'))
    snippets: List[str] = []
    picked = 0
    try:
        with DDGS() as ddgs:
            for t in terms[:max_terms]:
                try:
                    # Safe search params; we’ll grab short summaries
                    results = ddgs.text(t, max_results=per_term)
                    for r in results or []:
                        txt = (r.get('body') or r.get('title') or '')
                        txt = _space_re.sub(' ', txt).strip()
                        if txt:
                            snippets.append(f"{t}: {txt[:240]}")
                    picked += 1
                    if picked >= max_terms:
                        break
                except Exception:
                    continue
        # Bound overall time by just returning what we have quickly
    except Exception:
        pass
    return snippets[:max_terms * per_term]


# ---------------- Enrichment (cache + web optional) ----------------
_enrich_cache_path = 'artifacts/enrich_cache.json'
_enrich_cache: Dict[str, str] = {}
if os.path.exists(_enrich_cache_path):
    try:
        _enrich_cache = json.load(open(_enrich_cache_path, 'r', encoding='utf-8'))
    except Exception:
        _enrich_cache = {}

SAFE_SOURCES = ['https://en.wikipedia.org/api/rest_v1/page/summary/']

def extract_rare_terms(cv_text: str, top_k: int = 12) -> List[str]:
    toks = tokenize(cv_text)
    cands = [t for t in toks if (any(ch.isdigit() for ch in t) or len(t) >= 6) and t not in _stopwords]
    seen, out = set(), []
    for t in cands:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= top_k:
            break
    return out

def wiki_summary(term: str, timeout_sec: float = 1.0) -> str:
    key = f'wik:{term}'
    if key in _enrich_cache:
        return _enrich_cache[key]
    try:
        import requests
        url = SAFE_SOURCES[0] + requests.utils.quote(term)
        r = requests.get(url, timeout=timeout_sec, headers={'User-Agent':'cv-analyzer/1.0'})
        if r.status_code == 200:
            data = r.json()
            summ = data.get('extract') or ''
            summ = _space_re.sub(' ', summ).strip()
            if summ:
                _enrich_cache[key] = summ[:400]
                try:
                    os.makedirs(os.path.dirname(_enrich_cache_path), exist_ok=True)
                    json.dump(_enrich_cache, open(_enrich_cache_path,'w',encoding='utf-8'))
                except Exception:
                    pass
                return _enrich_cache[key]
    except Exception:
        pass
    _enrich_cache[key] = ''
    return ''

def build_enriched_cv(cv_text: str, job_text: str) -> str:
    terms = extract_rare_terms(cv_text, top_k=10)
    # Blend wiki summaries and optional web snippets
    snippets = []
    for t in terms[:6]:
        s = wiki_summary(t)
        if s:
            snippets.append(f'{t}: {s}')
        if len(snippets) >= 5:
            break
    # Optional web enrichment for a few salient terms (add from both CV and job for breadth)
    if os.getenv('GOOGLE_WEB_ENABLED', '0') == '1':
        more_terms = list(set(terms + extract_rare_terms(job_text, top_k=6)))
        web_snips = web_enrich_terms(more_terms, max_terms=3, per_term=2)
        for ws in web_snips:
            if ws:
                snippets.append(ws[:280])
            if len(snippets) >= 8:
                break
    if not snippets:
        return cv_text
    return cv_text + '\n\n' + '\n'.join(snippets)


# ---------------- Hybrid inference (semantic + adaptive + small “self-train”) ----------------
def fit_vectorizers(docs: List[str]) -> Tuple[Any, Any, Any, Any]:
    # “Self-train” a little by fitting on background corpus + current docs
    base = BACKGROUND_CORPUS + docs
    word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.995)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
    Xw = word_vec.fit_transform(base)
    Xc = char_vec.fit_transform(base)
    return word_vec, char_vec, Xw, Xc  # Xw/Xc on base only (we’ll refit on docs below for shapes)

def analyze_hybrid(cv_text: str, job_text: str):
    if TfidfVectorizer is None or cosine_similarity is None:
        return 0, [], []
    cv_enriched = build_enriched_cv(cv_text, job_text)

    # Split and expand job units with aliases
    j_units_raw = split_units(job_text, cap=200)
    if not j_units_raw:
        return 0, [], []
    j_variants = [expand_unit_aliases(ju) for ju in j_units_raw]
    j_units = j_units_raw[:]

    # CV units
    c_units = split_units(cv_enriched, cap=900)
    if not c_units:
        return 0, [], []

    # Augment with noun-phrases to improve overlap
    j_docs = augment_with_np(j_units)
    c_docs = augment_with_np(c_units)
    docs = j_docs + c_docs

    # Small “self-train”: fit vectorizers on background + request docs to set vocab/weights
    try:
        word_vec, char_vec, _, _ = fit_vectorizers(docs)
        Xw = word_vec.fit_transform(docs)
        Xc = char_vec.fit_transform(docs)
    except Exception:
        # Fallback: simple fit on docs
        word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.99)
        char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
        try:
            Xw = word_vec.fit_transform(docs)
            Xc = char_vec.fit_transform(docs)
        except Exception:
            return 0, [], []

    Jw, Cw = Xw[:len(j_docs), :], Xw[len(j_docs):, :]
    Jc, Cc = Xc[:len(j_docs), :], Xc[len(j_docs):, :]
    Sw = cosine_similarity(Jw, Cw)
    Sc = cosine_similarity(Jc, Cc)
    S = 0.65*Sw + 0.35*Sc

    matched, missing = [], []
    pairs = []
    base_threshold = 0.32
    cv_norm = normalize_text(' '.join(c_units))

    for i, ju in enumerate(j_units):
        row = S[i]
        bi = int(row.argmax())
        bs = float(row[bi])

        relax = 0.0
        if len(ju) <= 30:
            relax += 0.02
        if len(j_variants[i]) > 1:
            relax += 0.02
        thr = max(0.25, base_threshold - relax)

        is_match = bs >= thr

        if not is_match:
            variants = j_variants[i]
            if any(v in cv_norm for v in variants):
                is_match = True
            elif keyword_present(cv_norm, normalize_text(ju)):
                is_match = True

        pairs.append({'job_unit': ju, 'cv_unit': c_units[bi], 'score': bs})
        if is_match:
            matched.append(ju)
        else:
            missing.append(ju)

    avg = sum(p['score'] for p in pairs)/max(1, len(pairs))
    score = int(round(max(0.0, min(1.0, avg))*100))
    return score, matched[:50], missing[:50]


# ---------------- PDF/text parsing ----------------
def parse_pdf_bytes(data: bytes) -> str:
    if not data:
        return ''
    text = ''
    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text as _pdf_extract
        tmp_path = '_upload_tmp.pdf'
        with open(tmp_path, 'wb') as tmp:
            tmp.write(data)
        try:
            text = _pdf_extract(tmp_path) or ''
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        text = ''
    # Fallback: PyPDF
    if not text:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for i, p in enumerate(reader.pages):
                try:
                    pages.append(p.extract_text() or '')
                except Exception:
                    pages.append('')
                if len(pages) >= 20:
                    break
            text = '\n'.join([t for t in pages if t])
        except Exception:
            text = ''
    # Fallback: decode as text if enough readable content
    if not text:
        try:
            guess = data.decode('utf-8', errors='ignore')
            if len(re.findall(r'[A-Za-z]{2,}', guess)) > 50:
                text = guess
        except Exception:
            pass
    return text or ''


# ---------------- UI ----------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
    pre, textarea.cv { white-space:pre-wrap; background:#0f1524; padding:10px; border:1px solid var(--border); border-radius:10px; max-height:240px; overflow:auto; width:100%; min-height:120px; color:var(--text); }
    .warn { color:#f59e0b; font-size:12px; }
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
    <div class="brand">Bold <span class="accent">CV Analyzer</span></div>
  </header>
  <div class="container">
    <section class="panel" id="cv-panel">
      <h2>CV (PDF)</h2>
      <label for="pdf">Upload your CV</label>
      <input id="pdf" name="file" type="file" accept=".pdf" />
      <div id="pdf-status" class="status"></div>
      <label for="pdf-text">Extracted text (preview)</label>
      <pre id="pdf-text"></pre>
      <div class="warn">If your PDF is scanned or the preview is empty, paste your CV text below and still click Analyze.</div>
      <textarea id="cv-fallback" class="cv" placeholder="Paste your CV text here if preview is empty..."></textarea>
    </section>

    <section class="panel" id="job-panel">
      <h2>Job Requirements</h2>
      <label for="job">Paste the job description/requirements</label>
      <textarea id="job" placeholder="Paste job requirements here..."></textarea>
      <div class="actions">
        <button id="analyze">Analyze</button>
      </div>
      <div id="analyze-status" class="status"></div>
      <div id="results" style="display:none; margin-top:10px;">
        <div class="results">
          <div class="score" id="score">0</div>
          <div>
            <div><strong>Matched items</strong></div>
            <ul id="matched"></ul>
            <div style="margin-top:8px;"><strong>Missing items</strong></div>
            <ul id="missing"></ul>
            <div style="margin-top:8px;"><strong>Recommendations</strong></div>
            <ul id="recs"></ul>
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
    const input = el('pdf') || document.querySelector('input[type="file"][name="file"]');
    if(!input){ statusPDF('File input not found'); return; }
    input.accept = '.pdf';
    input.addEventListener('change', async function(){
      if(!input.files || !input.files[0]){ statusPDF('No file selected'); return; }
      const f = input.files[0]; if(!/\.pdf$/i.test(f.name)){ statusPDF('Please select a PDF'); return; }
      statusPDF('Parsing PDF...');
      try{
        const fd = new FormData(); fd.append('file', f);
        const resp = await fetch('/api/parse_pdf', {method:'POST', body:fd});
        const data = await resp.json();
        if(!data.ok){ statusPDF('Error: ' + (data.error || 'Unknown error')); return; }
        statusPDF('Parsed ' + (data.chars||0) + ' characters' + (data.warning? (' — ' + data.warning) : ''));
        el('pdf-text').textContent = data.text || '';
      } catch(e){ statusPDF('Upload failed: ' + e.message); }
    });
  }
  async function analyze(){
    const parsed = el('pdf-text').textContent || '';
    const fallback = el('cv-fallback').value || '';
    const cv = (parsed.trim()? parsed : fallback) || '';
    const job = el('job').value || '';
    if(!cv.trim()){ statusAnalyze('CV text is empty. Paste your CV text if PDF parsing failed.'); return; }
    if(!job.trim()){ statusAnalyze('Please paste the job requirements.'); return; }
    statusAnalyze('Analyzing...'); el('analyze').disabled = true;
    try{
      const resp = await fetch('/api/analyze', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ cv_text: cv, job_text: job })
      });
      const data = await resp.json();
      if(!data.ok){ statusAnalyze('Error: ' + (data.error || 'Unknown error')); return; }
      statusAnalyze(''); setScore(data.match_score || 0);
      const matched = el('matched'); matched.innerHTML='';
      const missing = el('missing'); missing.innerHTML='';
      const recs = el('recs'); recs.innerHTML='';
      (data.matched_keywords||[]).slice(0,50).forEach(k=>{ const li=document.createElement('li'); li.textContent = k; matched.appendChild(li); });
      (data.missing_keywords||[]).slice(0,50).forEach(k=>{ const li=document.createElement('li'); li.textContent = k; missing.appendChild(li); });
      (data.recommendations||[]).slice(0,10).forEach(r=>{ const li=document.createElement('li'); li.textContent = r; recs.appendChild(li); });
      el('results').style.display = '';
    } catch(e){ statusAnalyze('Analyze failed: ' + e.message); }
    finally { el('analyze').disabled = false; }
  }
  function bindAnalyze(){ el('analyze').addEventListener('click', analyze); }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', ()=>{ bindPDF(); bindAnalyze(); });
  } else { bindPDF(); bindAnalyze(); }
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
        resp = {'ok': True, 'text': preview, 'chars': len(text or '')}
        if not preview:
            resp['warning'] = 'Parsed 0 characters. If your PDF is scanned, paste CV text into the fallback box.'
        return jsonify(resp)
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

        recs = []
        if missing:
            recs.append('Consider adding evidence for: ' + ', '.join(missing[:8]) + '.')
        if score < 60:
            recs.append('Tailor your CV summary to mirror key role requirements and metrics.')
        cv_norm_once = normalize_text(cv)
        if 'experience' not in cv_norm_once:
            recs.append('Add a dedicated Experience section with impact-driven bullet points.')
        if 'project' not in cv_norm_once:
            recs.append('Include 1–2 relevant projects with outcomes and technologies used.')

        return jsonify({'ok': True,
                        'match_score': score,
                        'matched_keywords': matched,
                        'missing_keywords': missing,
                        'recommendations': recs})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
