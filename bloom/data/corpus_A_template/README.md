# Corpus A: template-based discrete mathematics questions

Source: a document of 500 numbered discrete mathematics questions without Bloom labels
(`500_Discrete_Math_Questions_No_Bloom_Labels.docx`). The question text was not modified.

## Files

| File | Rows | Purpose |
|---|---|---|
| `discrete_math_questions_full.csv` / `.jsonl` | 500 | all questions in the original order |
| `discrete_math_questions_unique.csv` / `.jsonl` | 48 | unique wordings only |

Encoding UTF-8; CSV is comma-separated; JSONL has one record per line.

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | str | identifier `DM001`…`DM500` |
| `source_number` | int | question number in the source document |
| `question` | str | question text (model input) |
| `topic` | str | discrete mathematics topic (16 values) |
| `template_id` | str | wording template `T1`…`T6` |
| `action_verb` | str | key task verb |
| `bloom_level` | str | Bloom level (**derived automatically** from the template, see below) |
| `bloom_level_id` | int | 1 = Remember … 6 = Create |
| `group_id` | str | `G01`…`G48`; identical wordings share a `group_id` |
| `is_duplicate` | 0/1 | 1 = the wording already appeared earlier |
| `n_copies` | int | how many times the wording occurs among the 500 questions |
| `split` | str | `train` / `validation` / `test` |

## How the Bloom labels were obtained

The source file has no labels. Every question follows one of six templates, and each template maps to
exactly one level:

| Template | Question opening | Level |
|---|---|---|
| T1 | State the definition… | Remember |
| T2 | Explain the main idea… | Understand |
| T3 | Solve a problem… | Apply |
| T4 | Analyze the structure… | Analyze |
| T5 | Determine whether… justify | Evaluate |
| T6 | Construct an original… | Create |

For blind labelling by experts or students, drop the columns `template_id`, `action_verb`,
`bloom_level` and `bloom_level_id`.

## Splits

The split is made **by group**, not by row: all copies of a wording go to the same split, otherwise test
questions would be verbatim copies of training questions. Within each Bloom level: 6 groups in train,
1 in validation, 1 in test (seed = 42).
Full file: train 375, validation 63, test 62 rows. Unique file: 36 / 6 / 6.

## Limitations

1. **Only 48 unique questions**, not 500: 6 templates × 16 topics repeat cyclically, each wording occurs 10–11 times.
2. **The label is determined by the first word.** Any classifier reaches about 100% accuracy by learning the verb,
   which says nothing about understanding the questions.
3. **Topic is confounded with level.** Eight topics occur only with Remember/Apply/Evaluate and the other eight
   only with Understand/Analyze/Create, so a model can also predict the level from the topic.
4. Questions are abstract (no concrete task content).

## Loading

```python
import pandas as pd
df = pd.read_csv("discrete_math_questions_unique.csv")
train = df[df.split == "train"]
test  = df[df.split == "test"]
X_train, y_train = train.question, train.bloom_level
```
