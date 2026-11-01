# -*- coding: utf-8 -*-
"""Step 3. Automatic Bloom-level classifier and its evaluation."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, json, re, warnings, pickle
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_val_predict, GroupKFold)
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)

OUT = "out/"
L = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}
RNG = 42

o = pd.read_csv(OUT + 'orig_audited.csv').rename(columns={'bloom_level_id': 'y'})
g = pd.read_csv(OUT + 'gen_audited.csv').rename(columns={'bloom_level_id': 'y'})

# ---------- first-word masking: does the model do more than read the verb? ----------
VERB1 = re.compile(r"^\s*\W*\w+")
mask_first_word = lambda q: VERB1.sub("<V>", q, count=1)
g['q_masked'] = g.question.map(mask_first_word)
o['q_masked'] = o.question.map(mask_first_word)


class HandFeats(BaseEstimator, TransformerMixin):
    """Interpretable features of the item wording."""
    PAT = {'digits': r"\d", 'math': r"[=∈⊆∪∩→↔∀∃⊕≡≤]",
           'given': r"\b(given|where|with|such that)\b", 'why': r"\bwhy\b",
           'whether': r"\bwhether\b", 'justify': r"\b(justify|defend|explain why)\b",
           'student_claim': r"\b(a student|claims|claim that|critique)\b",
           'compare': r"\b(compare|difference between|contrast)\b",
           'design': r"\b(design|devise|create|invent|formulate|propose|construct)\b",
           'own': r"\b(your own|original|real[- ]world|of your choice)\b",
           'list_semicolon': r";", 'question_mark': r"\?"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[len(q.split()), q.count(',')] +
                         [len(re.findall(p, q, re.I)) for p in self.PAT.values()]
                         for q in list(X)], dtype=float)


def pipe(kind):
    word = ('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2))
    char = ('c', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    hand = ('h', HandFeats())
    if kind == 'logreg':
        return Pipeline([('f', FeatureUnion([word, char, hand])),
                         ('m', LogisticRegression(max_iter=4000, C=5, class_weight='balanced'))])
    if kind == 'svm':
        return Pipeline([('f', FeatureUnion([word, char, hand])),
                         ('m', LinearSVC(C=1.0, class_weight='balanced', random_state=RNG))])
    if kind == 'nb':
        return Pipeline([('f', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                         ('m', ComplementNB())])
    if kind == 'rf':
        return Pipeline([('f', FeatureUnion([word, hand])),
                         ('m', RandomForestClassifier(n_estimators=600, random_state=RNG,
                                                      class_weight='balanced'))])
    if kind == 'words_only':
        return Pipeline([('f', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                         ('m', LogisticRegression(max_iter=4000, C=5, class_weight='balanced'))])


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {'acc': round(float(accuracy_score(y, p)), 4),
            'f1_macro': round(float(f1_score(y, p, average='macro')), 4),
            'qwk': round(float(cohen_kappa_score(y, p, weights='quadratic')), 4),
            'kappa': round(float(cohen_kappa_score(y, p)), 4),
            'adj_acc': round(float((np.abs(y - p) <= 1).mean()), 4),
            'mae': round(float(np.abs(y - p).mean()), 4)}


res = {}

# ---------- 3.1 baselines ----------
def baseline_firstword(tr, te):
    mp = tr.assign(fw=tr.question.str.split().str[0].str.lower()).groupby('fw').y.agg(lambda s: s.mode()[0])
    maj = tr.y.mode()[0]
    return te.question.str.split().str[0].str.lower().map(mp).fillna(maj).astype(int).values


cvb = StratifiedKFold(5, shuffle=True, random_state=RNG)
pb = np.zeros(len(g), dtype=int)
for tri, tei in cvb.split(g, g.y):
    pb[tei] = baseline_firstword(g.iloc[tri], g.iloc[tei])
res['baseline_firstword_gen_cv'] = metrics(g.y, pb)
res['baseline_random_gen'] = {'acc': round(1 / 6, 4)}

# ---------- 3.2 models on the generated bank, 5x5 stratified CV ----------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
for kind in ['words_only', 'nb', 'rf', 'svm', 'logreg']:
    accs, f1s, qwks = [], [], []
    for tri, tei in cv.split(g.question, g.y):
        m = pipe(kind).fit(g.question.iloc[tri], g.y.iloc[tri])
        p = m.predict(g.question.iloc[tei])
        accs.append(accuracy_score(g.y.iloc[tei], p))
        f1s.append(f1_score(g.y.iloc[tei], p, average='macro'))
        qwks.append(cohen_kappa_score(g.y.iloc[tei], p, weights='quadratic'))
    res['gen_cv_' + kind] = {'acc_mean': round(float(np.mean(accs)), 4),
                             'acc_sd': round(float(np.std(accs)), 4),
                             'acc_ci95': [round(float(np.percentile(accs, 2.5)), 4),
                                          round(float(np.percentile(accs, 97.5)), 4)],
                             'f1_macro': round(float(np.mean(f1s)), 4),
                             'qwk': round(float(np.mean(qwks)), 4)}

BEST = 'svm'
skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
oof = cross_val_predict(pipe(BEST), g.question, g.y, cv=skf)
res['gen_oof_best'] = metrics(g.y, oof)
res['gen_oof_confusion'] = confusion_matrix(g.y, oof, labels=range(1, 7)).tolist()
res['gen_oof_report'] = classification_report(g.y, oof, target_names=[L[i] for i in range(1, 7)], digits=3)
np.save(OUT + 'oof_pred.npy', oof)

# ---------- 3.3 ablation: first-word masking ----------
oof_m = cross_val_predict(pipe(BEST), g.q_masked, g.y, cv=skf)
res['gen_oof_masked_verb'] = metrics(g.y, oof_m)
np.save(OUT + 'oof_pred_masked.npy', oof_m)

# ---------- 3.4 training on audited labels M ----------
oof_M = cross_val_predict(pipe(BEST), g.question, g.M, cv=StratifiedKFold(5, shuffle=True, random_state=RNG))
res['gen_oof_trained_on_audited_M'] = metrics(g.M, oof_M)

# ---------- 3.5 template bank: grouped vs random CV ----------
p_o = cross_val_predict(pipe(BEST), o.question, o.y, cv=GroupKFold(n_splits=6), groups=o.group_id)
res['orig_groupcv_best'] = metrics(o.y, p_o)
p_o_naive = cross_val_predict(pipe(BEST), o.question, o.y, cv=skf)
res['orig_randomcv_best_LEAKY'] = metrics(o.y, p_o_naive)

# ---------- 3.6 cross-corpus transfer ----------
m_g = pipe(BEST).fit(g.question, g.y)
res['transfer_gen2orig'] = metrics(o.y, m_g.predict(o.question))
m_o = pipe(BEST).fit(o.question, o.y)
res['transfer_orig2gen'] = metrics(g.y, m_o.predict(g.question))
res['transfer_orig2gen_confusion'] = confusion_matrix(g.y, m_o.predict(g.question), labels=range(1, 7)).tolist()
m_gM = pipe(BEST).fit(g.question, g.M)
res['transfer_genM2orig'] = metrics(o.y, m_gM.predict(o.question))

# ---------- 3.7 final model ----------
allq = pd.concat([g[['question', 'y']], o[['question', 'y']]], ignore_index=True)
final = pipe(BEST).fit(allq.question, allq.y)
pickle.dump(final, open(OUT + 'bloom_model.pkl', 'wb'))
res['final_model'] = {'trained_on': len(allq), 'algo': BEST}

json.dump(res, open(OUT + '03_model.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print("=== STEP 3: CLASSIFIER ===\n")
print("-- baselines (LLM bank, 5-fold CV) --")
b = res['baseline_firstword_gen_cv']
print("  uniform prior               acc=0.167")
print(f"  first-word rule             acc={b['acc']:.3f}  F1={b['f1_macro']:.3f}  QWK={b['qwk']:.3f}")
print("\n-- models (LLM bank, 5x5 stratified CV) --")
print(f"  {'model':<14}{'acc':>26}{'F1 macro':>10}{'QWK':>8}")
for k in ['words_only', 'nb', 'rf', 'svm', 'logreg']:
    r = res['gen_cv_' + k]
    a = f"{r['acc_mean']:.3f} +-{r['acc_sd']:.3f} [{r['acc_ci95'][0]:.2f};{r['acc_ci95'][1]:.2f}]"
    print(f"  {k:<14}{a:>26}{r['f1_macro']:>10.3f}{r['qwk']:>8.3f}")
print("\n-- best model: TF-IDF (words+chars) + hand features -> linear SVM --")
r = res['gen_oof_best']
print(f"  acc={r['acc']:.3f}  F1={r['f1_macro']:.3f}  QWK={r['qwk']:.3f}  within±1={r['adj_acc']:.3f}  MAE={r['mae']:.3f}")
print(res['gen_oof_report'])
print("  confusion matrix (row = true, column = predicted):")
print(pd.DataFrame(res['gen_oof_confusion'], index=[L[i] for i in range(1, 7)],
                   columns=[L[i] for i in range(1, 7)]).to_string())
print("\n-- ablations and transfer --")
for k, t in [('gen_oof_masked_verb', 'LLM bank, first word masked'),
             ('gen_oof_trained_on_audited_M', 'LLM bank, trained on audited labels M'),
             ('orig_randomcv_best_LEAKY', 'Template 500, random CV (wording LEAKAGE)'),
             ('orig_groupcv_best', 'Template 500, grouped CV (no leakage)'),
             ('transfer_gen2orig', 'Transfer: train LLM 480 -> test template 500'),
             ('transfer_orig2gen', 'Transfer: train template 500 -> test LLM 480'),
             ('transfer_genM2orig', 'Transfer: train LLM 480 (labels M) -> template 500')]:
    r = res[k]
    print(f"  {t:<52} acc={r['acc']:.3f}  F1={r['f1_macro']:.3f}  QWK={r['qwk']:.3f}  ±1={r['adj_acc']:.3f}")
