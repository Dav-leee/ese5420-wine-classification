"""Reproduce the Wine checkpoint. Run: python run_baseline.py

Requires numpy, pandas, scipy, scikit-learn, matplotlib, threadpoolctl.
Place wine.zip here or one directory above; otherwise download from UCI.
All settings are fixed before inspecting scores. No seed selection.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import warnings

ROOT = Path(__file__).resolve().parent
SEED, NOISE_SEED, BOOT_SEED = 5420, 5421, 5422
KS = list(range(1, 32, 2))
CS = np.logspace(-3, 3, 7).tolist()
NOISE_COUNTS = [0, 10, 50, 100]
REPEATS = 20
FEATURES = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
            'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline']


def main():
    archive = ROOT / 'wine.zip'
    if not archive.exists():
        archive = ROOT.parent / 'wine.zip'
    if not archive.exists():
        archive = ROOT / 'wine.zip'
        urlretrieve('https://archive.ics.uci.edu/static/public/109/wine.zip', archive)
    raw = ZipFile(archive).read('wine.data')
    frame = pd.read_csv(io.BytesIO(raw), header=None, names=['class'] + FEATURES)
    X = frame[FEATURES].to_numpy(float)
    y = frame['class'].to_numpy(int)
    assert X.shape == (178, 13) and np.isfinite(X).all()
    assert set(y) == {1, 2, 3}
    frame.to_csv(ROOT / 'wine_loaded.csv', index=False)
    noise = np.random.default_rng(NOISE_SEED).normal(size=(len(y), 100))
    # Nested noise columns: the first 10 are identical in the 10/50/100 conditions.
    np.save(ROOT / 'noise_features.npy', noise)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS,
                                        random_state=SEED).split(X, y))
    pd.DataFrame([{'outer_fold': s, 'repeat': s//5, 'sample_id': int(i)}
                  for s, (_, test) in enumerate(splits) for i in test]).to_csv(
                      ROOT / 'outer_test_assignments.csv', index=False)
    # Resample wine IDs within class. Keep the 20 predictions of a wine together.
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_ids = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                   size=(10000, np.sum(y == c)), replace=True)
                                    for c in np.unique(y)], axis=1)
    summaries, fold_rows, oof_rows = [], [], []
    with threadpool_limits(limits=1):
        for q in NOISE_COUNTS:
            Z = np.column_stack([X, noise[:, :q]])
            models = ['kNN_scaled', 'logistic_scaled']
            if q == 0:
                models += ['kNN_unscaled']
            for name in models:
                predictions = np.full((REPEATS, len(y)), -1, dtype=int)
                chosen = []
                for s, (train, test) in enumerate(splits):
                    inner = StratifiedKFold(5, shuffle=True, random_state=SEED+1000+s)
                    if name.startswith('kNN'):
                        model = KNeighborsClassifier(algorithm='kd_tree', n_jobs=1)
                        grid = {'model__n_neighbors': KS}
                    else:
                        model = LogisticRegression(solver='lbfgs', max_iter=5000,
                                                   tol=1e-6, random_state=SEED)
                        grid = {'model__C': CS}
                    pipeline = Pipeline([('scale', StandardScaler() if name.endswith('_scaled')
                                          else 'passthrough'), ('model', model)])
                    search = GridSearchCV(pipeline, grid, cv=inner, scoring='accuracy',
                                          n_jobs=1, refit=True, error_score='raise')
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        search.fit(Z[train], y[train])
                    pred = search.predict(Z[test])
                    predictions[s//5, test] = pred
                    value = list(search.best_params_.values())[0]
                    chosen.append(value)
                    fold_rows.append({'noise': q, 'model': name, 'outer_fold': s,
                                      'repeat': s//5, 'accuracy': float(np.mean(pred == y[test])),
                                      'selected_parameter': value,
                                      'inner_best_accuracy': float(search.best_score_)})
                    for i, prediction in zip(test, pred):
                        oof_rows.append({'noise': q, 'model': name, 'repeat': s//5,
                                         'sample_id': int(i), 'truth': int(y[i]),
                                         'prediction': int(prediction)})
                    if (s+1) % 25 == 0:
                        print(f'noise={q:3d} {name:16s}: {s+1}/100 folds', flush=True)
                assert np.all(predictions != -1)
                correct = predictions == y[None, :]
                per_repeat = correct.mean(axis=1)
                per_wine = correct.mean(axis=0)
                boot = per_wine[bootstrap_ids].mean(axis=1)
                low, high = np.quantile(boot, [.025, .975])
                summaries.append({'noise': q, 'model': name,
                                  'accuracy': float(correct.mean()),
                                  'repeat_sd': float(per_repeat.std(ddof=1)),
                                  'bootstrap_low': float(low), 'bootstrap_high': float(high),
                                  'selected_counts': dict(Counter(map(str, chosen)))})
                print('RESULT', summaries[-1], flush=True)
                pd.DataFrame(summaries).to_csv(ROOT / 'summary.csv', index=False)
                pd.DataFrame(fold_rows).to_csv(ROOT / 'fold_results.csv', index=False)
                pd.DataFrame(oof_rows).to_csv(ROOT / 'out_of_fold_predictions.csv', index=False)
    metadata = {'N': len(y), 'features': 13, 'class_counts': dict(Counter(map(str, y))),
                'missing': int(frame.isna().sum().sum()),
                'duplicate_rows': int(frame.duplicated().sum()), 'dropped_rows': 0,
                'seed': SEED, 'noise_seed': NOISE_SEED, 'bootstrap_seed': BOOT_SEED,
                'k_grid': KS, 'C_grid': CS, 'repeats': REPEATS,
                'wine_data_sha256': hashlib.sha256(raw).hexdigest(),
                'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
                'interval': '95% class-stratified wine-ID percentile bootstrap; 10000 resamples; '
                            '20 held-out correctness values averaged per wine; no refitting. '
                            'Conditional descriptive interval, not a full training-sample uncertainty interval.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    draw(summaries)


def draw(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    colors = {'kNN_scaled': '#167b9a', 'logistic_scaled': '#c36b20', 'kNN_unscaled': '#666666'}
    labels = {'kNN_scaled': 'kNN + scaling', 'logistic_scaled': 'Logistic + scaling',
              'kNN_unscaled': 'kNN, no scaling'}
    base = [r for r in summaries if r['noise'] == 0]
    for i, r in enumerate(base):
        y = r['accuracy']*100
        axes[0].errorbar(i, y, yerr=[[y-r['bootstrap_low']*100],
                                     [r['bootstrap_high']*100-y]], fmt='o',
                        color=colors[r['model']], capsize=5, markersize=7)
        axes[0].annotate(f'{y:.2f}%', (i, r['bootstrap_high']*100), xytext=(0, 8),
                         textcoords='offset points', ha='center', fontsize=10)
    axes[0].set_xticks(range(3), ['kNN\nscaled', 'Logistic\nscaled', 'kNN\nunscaled'])
    axes[0].set_xlim(-.5, 2.5)
    axes[0].set_title('Original 13 chemical features')
    for name in ['kNN_scaled', 'logistic_scaled']:
        rows = [r for r in summaries if r['model'] == name]
        ys = np.array([r['accuracy']*100 for r in rows])
        axes[1].errorbar([r['noise'] for r in rows], ys,
                        yerr=[ys-[r['bootstrap_low']*100 for r in rows],
                              [r['bootstrap_high']*100 for r in rows]-ys],
                        fmt='o-', capsize=4, color=colors[name], label=labels[name])
    axes[1].set_xticks(NOISE_COUNTS)
    axes[1].set_xlabel('Added independent Gaussian noise features')
    axes[1].set_title('Irrelevant-feature stress test')
    axes[1].legend(loc='lower left', frameon=False)
    for ax in axes:
        ax.set_ylabel('Out-of-fold accuracy (%)')
        ax.set_ylim(60, 103)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Wine cultivar classification | nested 5-fold CV, 20 repeats', fontsize=14)
    fig.supxlabel('Bars: conditional 95% wine-level bootstrap intervals (10,000 resamples); no refitting.', fontsize=9)
    fig.savefig(ROOT / 'baseline.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
