# -*- coding: utf-8 -*-
"""Step 4. Statistical analysis: significance, confidence intervals, annotator agreement."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.miscmodels.ordinal_model import OrderedModel

OUT = "out/"
L = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}
RNG = np.random.default_rng(42)

g = pd.read_csv(OUT + 'gen_audited.csv').rename(columns={'bloom_level_id': 'y'})
o = pd.read_csv(OUT + 'orig_audited.csv').rename(columns={'bloom_level_id': 'y'})
oof = np.load(OUT + 'oof_pred.npy')
oof_m = np.load(OUT + 'oof_pred_masked.npy')
mdl = json.load(open(OUT + '03_model.json', encoding='utf-8'))
S = {}


def boot_ci(fn, *arrays, n=5000, alpha=0.05):
    """Percentile bootstrap CI for an arbitrary metric."""
    arrays = [np.asarray(a) for a in arrays]
    N = len(arrays[0])
    vals = [fn(*[a[i] for a in arrays]) for i in
            (RNG.integers(0, N, N) for _ in range(n))]
    return (round(float(np.percentile(vals, 100 * alpha / 2)), 4),
            round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 4))


# ========== 4.1 Agreement of three label sources: declared label, audit M, rubric R ==========
raters = g[['y', 'M', 'R']].values
counts, _ = aggregate_raters(raters, n_cat=7)
fk = float(fleiss_kappa(counts[:, 1:], method='fleiss'))
pairs = {}
for a, b in [('y', 'M'), ('y', 'R'), ('M', 'R')]:
    k = float(cohen_kappa_score(g[a], g[b]))
    kw = float(cohen_kappa_score(g[a], g[b], weights='quadratic'))
    ci = boot_ci(lambda x, yy: cohen_kappa_score(x, yy), g[a].values, g[b].values, n=2000)
    pairs[f'{a}_vs_{b}'] = {'kappa': round(k, 4), 'kappa_w': round(kw, 4), 'kappa_ci95': list(ci),
                            'agreement': round(float((g[a] == g[b]).mean()), 4)}
S['rater_agreement'] = {'fleiss_kappa_3_raters': round(fk, 4), 'pairwise': pairs}

# ========== 4.2 Is level inflation in the generated bank significant? ==========
dev = (g.M - g.y).values
n_over, n_under, n_dev = int((dev < 0).sum()), int((dev > 0).sum()), int((dev != 0).sum())
# sign test: are overstatements more frequent than understatements?
sign_p = float(stats.binomtest(n_over, n_over + n_under, 0.5, alternative='greater').pvalue)
# overall label error rate, Wilson CI
lo, hi = proportion_confint(n_dev, len(g), method='wilson')
S['label_error'] = {'n_mismatch': n_dev, 'rate': round(n_dev / len(g), 4),
                    'ci95': [round(float(lo), 4), round(float(hi), 4)],
                    'n_overstated': n_over, 'n_understated': n_under,
                    'sign_test_p': float(f"{sign_p:.3g}"),
                    'mean_signed_bias': round(float(dev.mean()), 4),
                    'wilcoxon_p': float(f"{stats.wilcoxon(dev[dev != 0]).pvalue:.3g}")}

# label errors by level: each level vs the pooled rest (Benjamini-Hochberg adjustment)
rows = []
for lv in range(1, 7):
    mask = (g.y == lv).values
    k1, n1 = int((dev[mask] != 0).sum()), int(mask.sum())
    k2, n2 = int((dev[~mask] != 0).sum()), int((~mask).sum())
    z, p = proportions_ztest([k1, k2], [n1, n2])
    lo, hi = proportion_confint(k1, n1, method='wilson')
    rows.append({'level': L[lv], 'err_rate': round(k1 / n1, 4),
                 'ci95': [round(float(lo), 4), round(float(hi), 4)], 'p_raw': float(p)})
p_adj = stats.false_discovery_control([r['p_raw'] for r in rows], method='bh')
for r, pa in zip(rows, p_adj):
    r['p_bh'] = float(f"{pa:.3g}"); r['p_raw'] = float(f"{r['p_raw']:.3g}")
S['label_error_by_level'] = rows

# ========== 4.3 Model vs baseline: McNemar test ==========
base = np.zeros(len(g), dtype=int)
from sklearn.model_selection import StratifiedKFold
cvb = StratifiedKFold(5, shuffle=True, random_state=42)
for tri, tei in cvb.split(g, g.y):
    tr = g.iloc[tri]
    mp = tr.assign(fw=tr.question.str.split().str[0].str.lower()).groupby('fw').y.agg(lambda s: s.mode()[0])
    base[tei] = g.iloc[tei].question.str.split().str[0].str.lower().map(mp).fillna(tr.y.mode()[0]).astype(int).values


def mcnemar_pair(a_ok, b_ok, name):
    t = np.array([[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
                  [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]])
    r = mcnemar(t, exact=True)
    return {'name': name, 'table': t.tolist(), 'statistic': float(r.statistic),
            'p': float(f"{r.pvalue:.3g}"),
            'acc_a': round(float(a_ok.mean()), 4), 'acc_b': round(float(b_ok.mean()), 4)}


ok_model = (oof == g.y.values); ok_base = (base == g.y.values); ok_mask = (oof_m == g.y.values)
S['mcnemar'] = [
    mcnemar_pair(ok_model, ok_base, 'SVM vs first-word rule'),
    mcnemar_pair(ok_model, ok_mask, 'SVM vs SVM with first word masked'),
]
S['model_ci'] = {
    'acc': round(float(ok_model.mean()), 4),
    'acc_ci95': list(proportion_confint(int(ok_model.sum()), len(g), method='wilson')),
    'acc_ci95_boot': list(boot_ci(accuracy_score, g.y.values, oof)),
    'qwk_ci95_boot': list(boot_ci(lambda a, b: cohen_kappa_score(a, b, weights='quadratic'), g.y.values, oof)),
    'f1_ci95_boot': list(boot_ci(lambda a, b: f1_score(a, b, average='macro'), g.y.values, oof)),
    'baseline_acc': round(float(ok_base.mean()), 4),
    'baseline_ci95': list(proportion_confint(int(ok_base.sum()), len(g), method='wilson')),
}
S['model_ci']['acc_ci95'] = [round(float(x), 4) for x in S['model_ci']['acc_ci95']]
S['model_ci']['baseline_ci95'] = [round(float(x), 4) for x in S['model_ci']['baseline_ci95']]

# ========== 4.4 Cross-corpus transfer: significance of the gap ==========
t1 = mdl['transfer_gen2orig']['acc']; t2 = mdl['transfer_orig2gen']['acc']
z, p = proportions_ztest([int(t1 * 500), int(t2 * 480)], [500, 480])
S['transfer_gap'] = {'gen2orig': t1, 'orig2gen': t2, 'z': round(float(z), 3), 'p': float(f"{p:.3g}"),
                     'cv_on_gen': mdl['gen_oof_best']['acc'],
                     'drop_cv_to_transfer': round(mdl['gen_oof_best']['acc'] - t2, 4)}

# ========== 4.5 Ordinal regression: which wording features predict the level ==========
X = pd.DataFrame({
    'n_words': g.question.str.split().str.len(),
    'has_digits': g.question.str.contains(r"\d").astype(int),
    'has_math': g.question.str.contains(r"[=∈⊆∪∩→↔∀∃⊕≡≤]").astype(int),
    'has_why': g.question.str.lower().str.contains(r"\bwhy\b").astype(int),
    'has_justify': g.question.str.lower().str.contains(r"\b(justify|defend|critique)\b").astype(int),
    'has_design': g.question.str.lower().str.contains(r"\b(design|devise|create|invent|formulate|construct)\b").astype(int),
})
X['n_words'] = (X.n_words - X.n_words.mean()) / X.n_words.std()
om = OrderedModel(g.M.values, X, distr='logit').fit(method='bfgs', disp=False)
S['ordinal_model'] = {
    'pseudo_r2': round(float(om.prsquared), 4), 'llr_p': float(f"{om.llr_pvalue:.3g}"),
    'coef': {k: {'beta': round(float(v), 3), 'p': float(f"{om.pvalues[k]:.3g}"),
                 'OR': round(float(np.exp(v)), 3)}
             for k, v in om.params.items() if k in X.columns}
}

# ========== 4.6 Template bank: effective sample size ==========
S['orig_dataset'] = {
    'effective_n': int(o.question.nunique()),
    'nominal_n': len(o),
    'design_effect_note': 'each wording is repeated 10-11 times; 48 independent observations, not 500',
    'cv_accuracy_group': mdl['orig_groupcv_best']['acc'],
    'cv_accuracy_random': mdl['orig_randomcv_best_LEAKY']['acc'],
    'underspecified_share': 0.498,
    # CI for 100% accuracy with n=48 independent wordings (rule of three)
    'acc_ci95_at_effective_n': [round(float(x), 4) for x in proportion_confint(48, 48, method='wilson')],
}

# ========== 4.7 Learning curve (how many labelled items are needed) ==========
from sklearn.model_selection import learning_curve
import importlib.util
spec = importlib.util.spec_from_file_location("m3", "03_model.py")
sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re as _re
from sklearn.base import BaseEstimator, TransformerMixin


class HF(BaseEstimator, TransformerMixin):
    PAT = {'d': r"\d", 'm': r"[=∈⊆∪∩→↔∀∃⊕≡≤]", 'w': r"\bwhy\b", 'j': r"\b(justify|defend|critique)\b",
           'g': r"\b(design|devise|create|formulate|construct)\b", 's': r";"}
    def fit(self, X, y=None): return self
    def transform(self, X): return np.array([[len(q.split())] + [len(_re.findall(p, q, _re.I)) for p in self.PAT.values()] for q in list(X)], float)


P = Pipeline([('f', FeatureUnion([('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                                  ('c', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
                                  ('h', HF())])),
              ('m', LinearSVC(C=1.0, class_weight='balanced', random_state=42))])
ns, tr_s, te_s = learning_curve(P, g.question, g.y, train_sizes=sizes,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1)
S['learning_curve'] = {'train_sizes': [int(x) for x in ns],
                       'test_acc_mean': [round(float(x), 4) for x in te_s.mean(1)],
                       'test_acc_sd': [round(float(x), 4) for x in te_s.std(1)]}

json.dump(S, open(OUT + '04_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ================= output =================
print("=== STEP 4: STATISTICAL ANALYSIS ===\n")
print("-- 4.1 Agreement of three label sources (declared / audit M / rubric R) --")
print(f"  Fleiss kappa (3 raters): {S['rater_agreement']['fleiss_kappa_3_raters']:.3f}")
for k, v in pairs.items():
    print(f"  {k:<10} agreement={v['agreement']:.1%}  kappa={v['kappa']:.3f} "
          f"95%CI[{v['kappa_ci95'][0]:.3f};{v['kappa_ci95'][1]:.3f}]  kappa_w={v['kappa_w']:.3f}")

print("\n-- 4.2 Systematic level inflation in the LLM bank --")
e = S['label_error']
print(f"  mismatches with audit: {e['n_mismatch']}/480 = {e['rate']:.1%}  95%CI[{e['ci95'][0]:.1%};{e['ci95'][1]:.1%}]")
print(f"  overstated {e['n_overstated']}, understated {e['n_understated']}; sign test p={e['sign_test_p']}")
print(f"  mean label bias: {e['mean_signed_bias']:+.3f} levels; Wilcoxon p={e['wilcoxon_p']}")
print(f"  {'level':<12}{'error rate':>13}{'95% CI':>20}{'p (BH)':>10}")
for r in rows:
    print(f"  {r['level']:<12}{r['err_rate']:>12.1%}  [{r['ci95'][0]:.2f}; {r['ci95'][1]:.2f}]{'':>6}{r['p_bh']:>10}")

print("\n-- 4.3 Model vs baseline (McNemar test, paired OOF predictions) --")
for m in S['mcnemar']:
    print(f"  {m['name']}")
    print(f"    accuracy {m['acc_a']:.3f} vs {m['acc_b']:.3f}; discordant "
          f"{m['table'][0][1]}/{m['table'][1][0]}; p={m['p']}")
c = S['model_ci']
print(f"  Model: acc={c['acc']:.3f} 95%CI[{c['acc_ci95'][0]:.3f};{c['acc_ci95'][1]:.3f}], "
      f"QWK 95%CI[{c['qwk_ci95_boot'][0]:.3f};{c['qwk_ci95_boot'][1]:.3f}]")
print(f"  Baseline: acc={c['baseline_acc']:.3f} 95%CI[{c['baseline_ci95'][0]:.3f};{c['baseline_ci95'][1]:.3f}]")

print("\n-- 4.4 Cross-corpus transfer --")
t = S['transfer_gap']
print(f"  LLM->template: {t['gen2orig']:.3f}   template->LLM: {t['orig2gen']:.3f}   z={t['z']}, p={t['p']}")
print(f"  drop CV({t['cv_on_gen']:.3f}) -> transfer({t['orig2gen']:.3f}): {t['drop_cv_to_transfer']:.3f}")

print("\n-- 4.5 Ordinal logit regression of level on wording features --")
om_ = S['ordinal_model']
print(f"  pseudo-R2={om_['pseudo_r2']:.3f}, model p={om_['llr_p']}")
print(f"  {'feature':<14}{'beta':>8}{'OR':>9}{'p':>12}")
for k, v in om_['coef'].items():
    print(f"  {k:<14}{v['beta']:>8.3f}{v['OR']:>9.3f}{v['p']:>12}")

print("\n-- 4.6 Template bank (500) --")
od = S['orig_dataset']
print(f"  nominal {od['nominal_n']} items, independent wordings {od['effective_n']}")
print(f"  model accuracy: random CV {od['cv_accuracy_random']:.3f} / grouped CV {od['cv_accuracy_group']:.3f}")
print(f"  with 48 independent observations and 100% accuracy, 95% CI = "
      f"[{od['acc_ci95_at_effective_n'][0]:.3f}; {od['acc_ci95_at_effective_n'][1]:.3f}]")

print("\n-- 4.7 Learning curve (LLM bank) --")
lc = S['learning_curve']
print("  " + "  ".join(f"n={a}:{b:.3f}" for a, b in zip(lc['train_sizes'], lc['test_acc_mean'])))
