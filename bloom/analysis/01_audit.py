# -*- coding: utf-8 -*-
"""Step 1. Does the actual cognitive demand match the declared Bloom level?"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, json, rubric, manual_audit as MA
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from scipy import stats
OUT = "out/"
L = rubric.LEVELS

orig = pd.read_csv('../data/corpus_A_template/discrete_math_questions_full.csv')
gen  = pd.read_csv('../data/corpus_C_generated/bloom_questions_generated.csv')

# ---------- annotator M (protocol-guided manual annotation) ----------
gen['M'] = gen.apply(lambda r: MA.DEVIATIONS.get(r.id, (r.bloom_level_id,""))[0], axis=1)
gen['M_note'] = gen.id.map(lambda i: MA.DEVIATIONS.get(i,(0,""))[1])
gen['borderline'] = gen.id.isin(MA.BORDERLINE).astype(int)
gen['M_strict'] = gen.apply(lambda r: MA.BORDERLINE.get(r.id, r.M), axis=1)
# ---------- annotator R (rule-based rubric) ----------
gen['R'] = gen.question.map(rubric.rubric_level)
orig['R'] = orig.question.map(rubric.rubric_level)

rep = {}
def kappa(a,b): return cohen_kappa_score(a,b,weights=None), cohen_kappa_score(a,b,weights='quadratic')

# === 1A. template bank (500) ===
orig['actionable'] = (~orig.template_id.isin(MA.UNDERSPECIFIED_TEMPLATES)).astype(int)
rep['orig'] = {
 'n': len(orig), 'unique_questions': int(orig.question.nunique()),
 'rubric_agreement': round(float((orig.R==orig.bloom_level_id).mean()),4),
 'kappa_R_vs_declared': round(float(cohen_kappa_score(orig.R,orig.bloom_level_id)),4),
 'underspecified_share': round(float(1-orig.actionable.mean()),4),
 'underspecified_templates': sorted(MA.UNDERSPECIFIED_TEMPLATES),
 'per_template': {t: {'declared': int(d.bloom_level_id.iloc[0]),
                      'rubric_mode': int(d.R.mode()[0]),
                      'actionable': bool(d.actionable.iloc[0]), 'n': len(d)}
                  for t,d in orig.groupby('template_id')},
}

# === 1B. generated bank (480) ===
k_md, kq_md = kappa(gen.M, gen.bloom_level_id)
k_rd, kq_rd = kappa(gen.R, gen.bloom_level_id)
k_rm, kq_rm = kappa(gen.R, gen.M)
k_sd, kq_sd = kappa(gen.M_strict, gen.bloom_level_id)
gen['dev'] = gen.M - gen.bloom_level_id
rep['gen'] = {
 'n': len(gen),
 'agree_M_declared': round(float((gen.M==gen.bloom_level_id).mean()),4),
 'agree_R_declared': round(float((gen.R==gen.bloom_level_id).mean()),4),
 'agree_R_M':        round(float((gen.R==gen.M).mean()),4),
 'kappa_M_declared': round(k_md,4), 'kappa_w_M_declared': round(kq_md,4),
 'kappa_R_declared': round(k_rd,4), 'kappa_w_R_declared': round(kq_rd,4),
 'kappa_R_M':        round(k_rm,4), 'kappa_w_R_M': round(kq_rm,4),
 'kappa_Mstrict_declared': round(k_sd,4), 'agree_Mstrict_declared': round(float((gen.M_strict==gen.bloom_level_id).mean()),4),
 'mean_signed_deviation': round(float(gen.dev.mean()),4),
 'n_overstated': int((gen.dev<0).sum()), 'n_understated': int((gen.dev>0).sum()),
 'borderline_share': round(float(gen.borderline.mean()),4),
}
# agreement by level
per_lvl = []
for lv in range(1,7):
    s = gen[gen.bloom_level_id==lv]
    per_lvl.append({'level': lv, 'name': L[lv], 'n': len(s),
        'agree_M': round(float((s.M==lv).mean()),3),
        'agree_R': round(float((s.R==lv).mean()),3),
        'inflation_M': round(float((s.M<lv).mean()),3),
        'mean_audited_level': round(float(s.M.mean()),2)})
rep['gen']['per_level'] = per_lvl

# Wilson binomial CIs for the agreement rate by level
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(float(c-h),3), round(float(c+h),3))
for row in per_lvl:
    s = gen[gen.bloom_level_id==row['level']]
    row['ci95_agree_M'] = wilson(int((s.M==row['level']).sum()), len(s))

# test: does level inflation depend on the declared level?
tab = pd.crosstab(gen.bloom_level_id, gen.dev!=0)
chi2,p,dof,_ = stats.chi2_contingency(tab)
rep['gen']['deviation_vs_level_chi2'] = {'chi2': round(chi2,3),'p': float(f"{p:.3g}"),'dof':int(dof),
    'cramers_v': round(float(np.sqrt(chi2/(len(gen)*(min(tab.shape)-1)))),3)}
# trend: does the overstatement rate grow with level? (Spearman correlation)
rho,prho = stats.spearmanr(gen.bloom_level_id, (gen.dev<0).astype(int))
rep['gen']['inflation_trend_spearman'] = {'rho': round(float(rho),3),'p': float(f"{prho:.3g}")}

cm_M = confusion_matrix(gen.bloom_level_id, gen.M, labels=range(1,7))
cm_R = confusion_matrix(gen.bloom_level_id, gen.R, labels=range(1,7))
rep['gen']['confusion_declared_vs_M'] = cm_M.tolist()
rep['gen']['confusion_declared_vs_R'] = cm_R.tolist()

gen.to_csv(OUT+'gen_audited.csv', index=False, encoding='utf-8')
orig.to_csv(OUT+'orig_audited.csv', index=False, encoding='utf-8')
json.dump(rep, open(OUT+'01_audit.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

print("=== STEP 1: LABEL AUDIT ===")
print(f"\n-- Template bank (500, {rep['orig']['unique_questions']} unique wordings) --")
for t,v in sorted(rep['orig']['per_template'].items()):
    print(f"  {t}: declared {L[v['declared']]:<10} rubric {L[v['rubric_mode']]:<10} "
          f"actionable={'yes' if v['actionable'] else 'NO'}  n={v['n']}")
print(f"  share of non-actionable items (no task content): {rep['orig']['underspecified_share']:.1%}")
print(f"\n-- Generated bank (480) --")
print(f"  agreement audit M vs declared       : {rep['gen']['agree_M_declared']:.1%}  kappa={k_md:.3f}  kappa_w={kq_md:.3f}")
print(f"  agreement rubric R vs declared      : {rep['gen']['agree_R_declared']:.1%}  kappa={k_rd:.3f}  kappa_w={kq_rd:.3f}")
print(f"  agreement R vs M (rubric validity)  : {rep['gen']['agree_R_M']:.1%}  kappa={k_rm:.3f}")
print(f"  strict reading of boundary cases    : {rep['gen']['agree_Mstrict_declared']:.1%}  kappa={k_sd:.3f}")
print(f"  overstated: {rep['gen']['n_overstated']}, understated: {rep['gen']['n_understated']}, "
      f"mean signed deviation: {rep['gen']['mean_signed_deviation']:+.3f}")
print("\n  level        n   agr.M   95% CI         inflation  mean audited level")
for r in per_lvl:
    print(f"  {r['name']:<11}{r['n']:>3}  {r['agree_M']:>6.1%}  {str(r['ci95_agree_M']):<14} {r['inflation_M']:>7.1%}   {r['mean_audited_level']:.2f}")
d=rep['gen']['deviation_vs_level_chi2']; t=rep['gen']['inflation_trend_spearman']
print(f"\n  chi2 (deviation ~ level): chi2={d['chi2']}, p={d['p']}, V={d['cramers_v']}")
print(f"  inflation trend by level  : rho={t['rho']}, p={t['p']}")
print("\n  Matrix (row = declared, column = audit M):")
print(pd.DataFrame(cm_M, index=[L[i] for i in range(1,7)], columns=[L[i] for i in range(1,7)]).to_string())
