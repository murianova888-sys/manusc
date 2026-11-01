# -*- coding: utf-8 -*-
"""Step 0. Corpus B (six per-level files) is Corpus A sorted by its existing labels."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, glob, json
a = pd.read_csv('../data/corpus_A_template/discrete_math_questions_full.csv')
b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('../data/corpus_B_per_level/*.csv'))],
              ignore_index=True)
m = a.merge(b, on='id', suffixes=('_A', '_B'))
rep = {'n_A': len(a), 'n_B': len(b), 'same_ids': set(a.id) == set(b.id),
       'question_agreement': round(float((m.question_A == m.question_B).mean()), 4),
       'label_agreement': round(float((m.bloom_level_id_A == m.bloom_level_id_B).mean()), 4)}
json.dump(rep, open('out/00_corpus_b.json', 'w', encoding='utf-8'), indent=2)
print(rep)
