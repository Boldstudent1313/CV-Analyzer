
import json, re, os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = Path('data')
ART_DIR = Path('artifacts')
ART_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / 'cv_examples.json', 'r', encoding='utf-8') as f:
    cvs = json.load(f)
with open(DATA_DIR / 'jd_examples.json', 'r', encoding='utf-8') as f:
    jds = json.load(f)

def split_units(text: str):
    parts = re.split(r"[\n\r]+|\.\s+|;\s+|\-\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]

pairs = []
for jd in jds:
    role = jd['role']
    jd_units = split_units(jd['text'])
    for cv in cvs:
        cv_units = split_units(cv['text'])
        label = 1 if (role == 'Data Scientist' and 'data scientist' in cv['text'].lower()) or \
                      (role == 'Backend Engineer' and 'backend engineer' in cv['text'].lower()) or \
                      (role == 'ML Engineer' and ('machine learning engineer' in cv['text'].lower() or 'ml ' in cv['text'].lower())) else 0
        for ju in jd_units:
            if label == 1:
                hit = False
                for cu in cv_units:
                    overlap = len(set(re.findall(r"[a-zA-Z0-9+#.]+", ju.lower())) & set(re.findall(r"[a-zA-Z0-9+#.]+", cu.lower())))
                    if overlap >= 2:
                        pairs.append((ju, cu, 1))
                        hit = True
                if not hit and cv_units:
                    pairs.append((ju, cv_units[0], 1))
            else:
                for cu in cv_units[:2]:
                    pairs.append((ju, cu, 0))

word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)

jd_texts = [p[0] for p in pairs]
cv_texts = [p[1] for p in pairs]
labels = np.array([p[2] for p in pairs], dtype=np.int64)

X_j_word = word_vec.fit_transform(jd_texts)
X_j_char = char_vec.fit_transform(jd_texts)
X_c_word = word_vec.transform(cv_texts)
X_c_char = char_vec.transform(cv_texts)

from sklearn.preprocessing import normalize
X_j_word = normalize(X_j_word)
X_c_word = normalize(X_c_word)
X_j_char = normalize(X_j_char)
X_c_char = normalize(X_c_char)

sim_word = np.sum(X_j_word.multiply(X_c_word), axis=1)
sim_char = np.sum(X_j_char.multiply(X_c_char), axis=1)

X = np.hstack([np.asarray(sim_word).reshape(-1,1), np.asarray(sim_char).reshape(-1,1)])

X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

val_proba = clf.predict_proba(X_val)[:,1]
try:
    auc = roc_auc_score(y_val, val_proba)
except Exception:
    auc = float('nan')
ap = average_precision_score(y_val, val_proba)
metrics = {'roc_auc': float(auc) if auc==auc else None, 'avg_precision': float(ap)}

joblib.dump(word_vec, ART_DIR / 'tfidf_word.joblib')
joblib.dump(char_vec, ART_DIR / 'tfidf_char.joblib')
joblib.dump(clf, ART_DIR / 'pair_clf.joblib')
with open(ART_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f)
with open(ART_DIR / 'config.json', 'w', encoding='utf-8') as f:
    json.dump({'threshold': 0.5}, f)
print('Training complete. Metrics:', metrics)
