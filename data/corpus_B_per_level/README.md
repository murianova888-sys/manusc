# Corpus B: per-level partition of Corpus A

Six CSV files, one per Bloom level (`1_Remember.csv` … `6_Create.csv`), with the same fields as Corpus A.
`analysis/00_check_corpus_b.py` verifies that Corpus B contains the same 500 item identifiers, texts and labels
as Corpus A (agreement 1.000), i.e. it is Corpus A sorted by its existing labels, not an independent annotation.
