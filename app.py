
from flask import Flask, render_template_string

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>CV Analyzer</title>
  <style>
    .strategy-b, #strategy-b { display: none !important; }
    body { font-family: sans-serif; margin: 20px; }
  </style>
</head>
<body>
  <h1>CV Analyzer</h1>
  <div class='strategy-a'>
    <label for='pdf'>Upload CV (PDF)</label>
    <input id='pdf' name='file' type='file' accept='.pdf' />
    <div id='pdf-status'></div>
    <pre id='pdf-text' style='white-space: pre-wrap;'></pre>
  </div>
  <script>
  (function(){
    function status(msg){
      var el = document.getElementById('pdf-status');
      if(!el){
        el = document.createElement('div'); el.id='pdf-status'; el.style.marginTop='8px';
        var p = document.getElementById('pdf');
        (p && p.parentNode ? p.parentNode : document.body).appendChild(el);
      }
      el.textContent = msg;
    }
    function showText(t){
      var el = document.getElementById('pdf-text');
      if(!el){
        el = document.createElement('pre'); el.id='pdf-text'; el.style.whiteSpace='pre-wrap'; el.style.marginTop='8px';
        var p = document.getElementById('pdf');
        (p && p.parentNode ? p.parentNode : document.body).appendChild(el);
      }
      el.textContent = t;
    }
    function bind(){
      var input = document.getElementById('pdf') || document.querySelector('input[type="file"][name="file"]');
      if(!input){ status('File input not found on page'); return; }
      input.accept = '.pdf';
      input.addEventListener('change', async function(){
        if(!input.files || !input.files[0]){ status('No file selected'); return; }
        var file = input.files[0];
        if(!/\.pdf$/i.test(file.name)){ status('Please select a PDF'); return; }
        status('Parsing PDF...');
        try{
          var fd = new FormData(); fd.append('file', file);
          const resp = await fetch('/api/parse_pdf', { method: 'POST', body: fd });
          let data; try { data = await resp.json(); } catch(e){ status('Invalid JSON from server'); return; }
          if(!data.ok){ status('Error: ' + (data.error || 'Unknown error')); return; }
          status('Parsed ' + (data.chars||0) + ' characters');
          showText(data.text || '');
        }catch(err){ status('Upload failed: ' + err.message); }
      });
    }
    if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind); else bind();
  })();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/health', methods=['GET'])
def health():
    return ('ok', 200)

@app.route('/api/parse_pdf', methods=['POST'])
def parse_pdf():
    from flask import request, jsonify
    import os
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file part'}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'ok': False, 'error': 'No selected file'}), 400
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'ok': False, 'error': 'Only PDF files are supported'}), 400
    try:
        data = f.read()
        try:
            from pdfminer.high_level import extract_text
        except Exception:
            extract_text = None
        if extract_text is None:
            return jsonify({'ok': False, 'error': 'PDF parser unavailable'}), 500
        tmp_path = '_upload_tmp.pdf'
        with open(tmp_path, 'wb') as tmp:
            tmp.write(data)
        try:
            text = extract_text(tmp_path) or ''
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        preview = text[:20000]
        return jsonify({'ok': True, 'text': preview, 'chars': len(text)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
