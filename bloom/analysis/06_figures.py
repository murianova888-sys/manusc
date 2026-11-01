# -*- coding: utf-8 -*-
"""Step 6. Figures for the paper (300 dpi, grayscale-safe)."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "out/"
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': .8,
    'axes.edgecolor': '#444444', 'xtick.color': '#444444', 'ytick.color': '#444444',
    'axes.labelcolor': '#111111', 'text.color': '#111111',
    'xtick.major.width': .8, 'ytick.major.width': .8, 'savefig.dpi': 300,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .04,
})
A = json.load(open(OUT + '01_audit.json', encoding='utf-8'))
S = json.load(open(OUT + '04_stats.json', encoding='utf-8'))
M = json.load(open(OUT + '03_model.json', encoding='utf-8'))
LV = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
BLUE, GREY, DARK = '#2a6fb5', '#9aa3ab', '#16324f'

# ---- Figure 1: share of labels disagreeing with the audit, by level, with Wilson CIs ----
rows = S['label_error_by_level']
fig, ax = plt.subplots(figsize=(5.6, 2.9))
y = np.arange(6)[::-1]
rate = [r['err_rate'] for r in rows]
lo = [r['err_rate'] - r['ci95'][0] for r in rows]
hi = [r['ci95'][1] - r['err_rate'] for r in rows]
sig = [r['p_bh'] < .05 and r['err_rate'] > .05 for r in rows]
cols = [DARK if s else BLUE for s in sig]
ax.barh(y, rate, height=.5, color=cols, zorder=3)
ax.errorbar(rate, y, xerr=[lo, hi], fmt='none', ecolor='#555555',
            elinewidth=.9, capsize=3, capthick=.9, zorder=4)
for i, r in enumerate(rows):
    ax.text(r['ci95'][1] + .012, y[i], f"{r['err_rate']*100:.1f}%", va='center', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(LV)
ax.set_xlim(0, .30); ax.set_xticks([0, .05, .10, .15, .20, .25, .30])
ax.set_xticklabels(['0', '5', '10', '15', '20', '25', '30%'])
ax.set_xlabel('Items whose declared level disagrees with the audit')
ax.grid(axis='x', color='#dddddd', lw=.6, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
fig.savefig(OUT + 'fig1_error_by_level.png'); plt.close(fig)

# ---- Figure 2: declared x audited matrix ----
cm = np.array(A['gen']['confusion_declared_vs_M'])
fig, ax = plt.subplots(figsize=(4.5, 4.0))
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cfe0f2', '#5d94c9', '#16324f'])
ax.imshow(np.sqrt(cm), cmap=cmap, vmin=0, vmax=np.sqrt(80))
for i in range(6):
    for j in range(6):
        v = cm[i, j]
        if v == 0: continue
        ax.text(j, i, str(v), ha='center', va='center', fontsize=8.5,
                color='white' if v > 35 else '#111111',
                fontweight='bold' if (i != j and v > 5) else 'normal')
for i, j in [(3, 2), (5, 2)]:
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='#b3261e', lw=1.6))
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels([l[:3] for l in LV]); ax.set_yticklabels(LV)
ax.set_xlabel('Audited level'); ax.set_ylabel('Declared level')
ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.grid(which='minor', color='#e6e6e6', lw=.7)
ax.tick_params(which='minor', length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(OUT + 'fig2_confusion.png'); plt.close(fig)

# ---- Figure 3: accuracy across evaluation conditions ----
conds = [
    ('Chance (1/6)', 1 / 6, GREY),
    ('First-word rule', M['baseline_firstword_gen_cv']['acc'], '#7aa8d4'),
    ('SVM, 5x5 CV', M['gen_oof_best']['acc'], BLUE),
    ('SVM, first word masked', M['gen_oof_masked_verb']['acc'], '#8fb4d9'),
    ('Template corpus, grouped CV', M['orig_groupcv_best']['acc'], DARK),
    ('Transfer: LLM -> template', M['transfer_gen2orig']['acc'], BLUE),
    ('Transfer: template -> LLM', M['transfer_orig2gen']['acc'], '#b3261e'),
]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
y = np.arange(len(conds))[::-1]
ax.barh(y, [c[1] for c in conds], height=.52, color=[c[2] for c in conds], zorder=3)
ax.axvline(M['baseline_firstword_gen_cv']['acc'], color='#b3261e', ls='--', lw=1, zorder=4)
for i, c in enumerate(conds):
    ax.text(c[1] + .012, y[i], f"{c[1]:.3f}", va='center', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([c[0] for c in conds])
ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1.0])
ax.set_xlabel('Accuracy (6-way cognitive-level classification)')
ax.grid(axis='x', color='#dddddd', lw=.6, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
fig.savefig(OUT + 'fig3_accuracy.png'); plt.close(fig)

# ---- Figure 4: learning curve ----
lc = S['learning_curve']
fig, ax = plt.subplots(figsize=(5.0, 2.7))
n = np.array(lc['train_sizes']); a = np.array(lc['test_acc_mean']); sd = np.array(lc['test_acc_sd'])
ax.fill_between(n, a - sd, a + sd, color=BLUE, alpha=.16, lw=0)
ax.plot(n, a, '-o', color=BLUE, lw=1.6, ms=4.5, mec='white', mew=.9, zorder=3)
ax.axhline(M['baseline_firstword_gen_cv']['acc'], color='#b3261e', ls='--', lw=1)
ax.text(n[-1], M['baseline_firstword_gen_cv']['acc'] + .022, 'first-word rule (0.917)',
        ha='right', fontsize=8, color='#b3261e')
ax.set_xlabel('Labelled items in the training set')
ax.set_ylabel('Held-out accuracy')
ax.set_ylim(.2, 1.0); ax.grid(color='#dddddd', lw=.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.savefig(OUT + 'fig4_learning_curve.png'); plt.close(fig)

print("Figures saved:")
for f in ['fig1_error_by_level.png', 'fig2_confusion.png', 'fig3_accuracy.png', 'fig4_learning_curve.png']:
    print("  out/" + f)
