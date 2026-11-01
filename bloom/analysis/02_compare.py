# -*- coding: utf-8 -*-
"""Step 2. Comparison of the template bank (500) and the LLM-generated bank (480)."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, json, re
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
OUT="out/"; L={1:"Remember",2:"Understand",3:"Apply",4:"Analyze",5:"Evaluate",6:"Create"}

o=pd.read_csv(OUT+'orig_audited.csv'); g=pd.read_csv(OUT+'gen_audited.csv')
o['first_word']=o.question.str.split().str[0].str.lower()
for d in (o,g): d['n_words']=d.question.str.split().str.len()
o=o.rename(columns={'bloom_level_id':'y'}); g=g.rename(columns={'bloom_level_id':'y'})

def leak_first_word(d):
    """Share of items whose level is recovered from the first word (modal class)."""
    mp=d.groupby('first_word').y.agg(lambda s:s.mode()[0])
    return float((d.first_word.map(mp)==d.y).mean())

def cond_entropy(d,col):
    """H(level | feature), bits. 0 = the feature fully determines the label."""
    h=0.0
    for v,s in d.groupby(col):
        p=s.y.value_counts(normalize=True)
        h+=len(s)/len(d)*(-(p*np.log2(p)).sum())
    return float(h)

def mutual_info_norm(d,col):
    H=-(d.y.value_counts(normalize=True)*np.log2(d.y.value_counts(normalize=True))).sum()
    return float((H-cond_entropy(d,col))/H)

def specificity(q):
    """Subject specificity cues: numbers, formulas, explicitly given objects."""
    return int(bool(re.search(r"\d", q)) or bool(re.search(r"[=∈⊆∪∩→↔∀∃⊕≡≤<>{}]", q))
               or bool(re.search(r"\b[A-Z]\(|\bf\(x\)|\ba[ₙ]", q)))

def inter_level_overlap(d):
    """Mean cosine similarity of TF-IDF centroids of different levels: higher = levels are lexically closer."""
    v=TfidfVectorizer(stop_words=None, min_df=1).fit_transform(d.question)
    cent=np.vstack([np.asarray(v[(d.y==k).values].mean(axis=0)) for k in range(1,7)])
    s=cosine_similarity(cent); iu=np.triu_indices(6,1)
    return float(s[iu].mean())

rows=[]
for name,d in [("Template 500",o),("LLM 480",g)]:
    verbs=Counter(d.first_word)
    rows.append({
      'dataset':name,'n':len(d),'unique_questions':int(d.question.nunique()),
      'unique_first_words':int(d.first_word.nunique()),
      'topic_level_cells_filled':int(d.groupby(['topic','y']).size().shape[0]),
      'topic_level_cells_total':int(d.topic.nunique()*6),
      'leak_first_word':round(leak_first_word(d),4),
      'leak_topic':round(float((d.groupby('topic').y.agg(lambda s:s.mode()[0]).reindex(d.topic).values==d.y.values).mean()),4),
      'NMI_first_word':round(mutual_info_norm(d,'first_word'),4),
      'NMI_topic':round(mutual_info_norm(d,'topic'),4),
      'mean_words':round(float(d.n_words.mean()),1),'sd_words':round(float(d.n_words.std()),1),
      'specificity':round(float(d.question.map(specificity).mean()),4),
      'type_token_ratio':round(float(len(set(" ".join(d.question).lower().split()))/len(" ".join(d.question).split())),4),
      'inter_level_tfidf_similarity':round(inter_level_overlap(d),4),
      'top5_first_words':dict(verbs.most_common(5)),
    })
cmp=pd.DataFrame(rows)

# length ~ level (ANOVA + Kruskal-Wallis) for the generated bank
groups=[g[g.y==k].n_words.values for k in range(1,7)]
F,pF=stats.f_oneway(*groups); H,pH=stats.kruskal(*groups)
eta2=float(sum(len(x)*(x.mean()-g.n_words.mean())**2 for x in groups)/((g.n_words-g.n_words.mean())**2).sum())
rho,prho=stats.spearmanr(g.y,g.n_words)
len_stats={'anova_F':round(float(F),3),'anova_p':float(f"{pF:.3g}"),'eta2':round(eta2,4),
           'kruskal_H':round(float(H),3),'kruskal_p':float(f"{pH:.3g}"),
           'spearman_rho':round(float(rho),3),'spearman_p':float(f"{prho:.3g}"),
           'mean_words_by_level':{L[k]:round(float(g[g.y==k].n_words.mean()),1) for k in range(1,7)}}

# test of independence of topic and level
for name,d in [("orig",o),("gen",g)]:
    tab=pd.crosstab(d.topic,d.y); chi2,p,dof,_=stats.chi2_contingency(tab)
    v=float(np.sqrt(chi2/(len(d)*(min(tab.shape)-1))))
    len_stats[f'topic_level_chi2_{name}']={'chi2':round(float(chi2),1),'p':float(f"{p:.3g}"),'cramers_v':round(v,3)}

json.dump({'comparison':rows,'length_analysis':len_stats},
          open(OUT+'02_compare.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

print("=== STEP 2: CORPUS COMPARISON ===\n")
show=['n','unique_questions','unique_first_words','topic_level_cells_filled','leak_first_word',
      'leak_topic','NMI_first_word','NMI_topic','mean_words','specificity','type_token_ratio',
      'inter_level_tfidf_similarity']
names={'n':'items','unique_questions':'unique wordings','unique_first_words':'distinct first words',
 'topic_level_cells_filled':'topic x level cells filled','leak_first_word':'leakage: level from first word',
 'leak_topic':'leakage: level from topic','NMI_first_word':'NMI first word <-> level',
 'NMI_topic':'NMI topic <-> level','mean_words':'mean length, words',
 'specificity':'share with concrete content','type_token_ratio':'lexical diversity (TTR)',
 'inter_level_tfidf_similarity':'lexical similarity of levels (TF-IDF)'}
print(f"{'measure':<42}{'Template 500':>14}{'LLM 480':>12}")
for k in show:
    a,b=cmp[k].values
    f=lambda x:f"{x:.3f}" if isinstance(x,float) else str(x)
    print(f"  {names[k]:<40}{f(a):>14}{f(b):>12}")
print("\n  first words (top 5):")
for r in rows: print(f"    {r['dataset']:<14}{r['top5_first_words']}")
print(f"\n  Item length vs level (LLM): ANOVA F={len_stats['anova_F']}, p={len_stats['anova_p']}, eta^2={len_stats['eta2']}")
print(f"  Spearman rho={len_stats['spearman_rho']}, p={len_stats['spearman_p']}")
print(f"  {len_stats['mean_words_by_level']}")
print(f"\n  Topic<->level dependence: template V={len_stats['topic_level_chi2_orig']['cramers_v']} "
      f"(p={len_stats['topic_level_chi2_orig']['p']}), LLM V={len_stats['topic_level_chi2_gen']['cramers_v']} "
      f"(p={len_stats['topic_level_chi2_gen']['p']})")
