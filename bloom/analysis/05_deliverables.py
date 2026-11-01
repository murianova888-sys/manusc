# -*- coding: utf-8 -*-
"""Step 5. Output files: audited dataset, relabelling list, combined corpus."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, json, numpy as np
OUT="out/"; L={1:"Remember",2:"Understand",3:"Apply",4:"Analyze",5:"Evaluate",6:"Create"}
g=pd.read_csv(OUT+'gen_audited.csv'); o=pd.read_csv(OUT+'orig_audited.csv')

# 1) generated bank with audit columns
cols=['id','question','topic','bloom_level','bloom_level_id','M','R','borderline','M_note','split']
d=g[cols].copy()
d['audit_level']=d.M; d['audit_level_name']=d.M.map(L)
d['rubric_level']=d.R; d['rubric_level_name']=d.R.map(L)
d['status']=np.where(d.M!=d.bloom_level_id,'relabel',
             np.where(d.borderline==1,'boundary case','confirmed'))
d[['id','question','topic','bloom_level','bloom_level_id','audit_level','audit_level_name',
   'rubric_level','rubric_level_name','status','M_note','split']].to_csv(
   OUT+'generated_with_audit.csv',index=False,encoding='utf-8-sig')

# 2) items to relabel
flag=d[d.M!=d.bloom_level_id][['id','question','bloom_level','audit_level_name','M_note']]
flag.columns=['id','question','declared_level','audited_level','justification']
flag.to_csv(OUT+'questions_to_relabel.csv',index=False,encoding='utf-8-sig')

# 3) template bank with actionability flag
o2=o[['id','question','topic','template_id','bloom_level','bloom_level_id','R','actionable']].copy()
o2['problem']=np.where(o2.actionable==0,'item contains no task content','')
o2.to_csv(OUT+'original_with_audit.csv',index=False,encoding='utf-8-sig')

# 4) combined training corpus
u=pd.concat([g[['question','topic','bloom_level_id','M']].assign(source='ai_generated'),
             o.drop_duplicates('question')[['question','topic','bloom_level_id']].assign(source='original_template',M=lambda x:x.bloom_level_id)],
            ignore_index=True)
u['bloom_level']=u.M.map(L)
u.rename(columns={'M':'gold_level_id'}).drop(columns=['bloom_level_id']).to_csv(
    OUT+'training_corpus.csv',index=False,encoding='utf-8-sig')

print("Output files:")
for f in ['generated_with_audit.csv','questions_to_relabel.csv','original_with_audit.csv','training_corpus.csv','bloom_model.pkl']:
    print("  out/"+f)
print(f"\nStatus counts in the generated bank: {d.status.value_counts().to_dict()}")
print(f"Training corpus: {len(u)} unique items")
print("\nItems to relabel (28):")
for _,r in flag.iterrows():
    print(f"  {r['id']}  {r['declared_level']:<10} -> {r['audited_level']:<10}  {r['question'][:62]}")
