# -*- coding: utf-8 -*-
"""Load the trained model (out/bloom_model.pkl) and predict Bloom levels.

The pickled pipeline references the HandFeats transformer that 03_model.py
defines at script level, so the class must exist in __main__ before unpickling.
This file repeats that class verbatim (it is not re-trained here).

Usage (from analysis/):
    python predict.py "Construct a truth table for (p -> q) and (q -> p)."
    python predict.py questions.txt          # one question per line
"""
import sys, re, pickle, os
import numpy as np
import __main__
from sklearn.base import BaseEstimator, TransformerMixin

LEVELS = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}


class HandFeats(BaseEstimator, TransformerMixin):
    """Interpretable features of the item wording (identical to 03_model.py)."""
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


__main__.HandFeats = HandFeats  # makes unpickling work when this file is imported


def load_model(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bloom_model.pkl")):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(questions, model=None):
    model = model or load_model()
    return [LEVELS[int(k)] for k in model.predict(list(questions))]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    qs = ([l.strip() for l in open(arg, encoding="utf-8") if l.strip()]
          if os.path.isfile(arg) else sys.argv[1:])
    for q, lvl in zip(qs, predict(qs)):
        print(f"{lvl:<11} {q}")
