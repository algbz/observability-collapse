"""
Empirical Analysis: Observability Collapse in ESM Data
======================================================
Tests whether pre-transition variance decreases (observability
collapse) or increases (standard critical slowing down) in
experience sampling data.

Results reported in Section 5 of the paper.

Usage:
    python analysis_openesm.py

Requires: pip install openesm pandas numpy scipy matplotlib
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
WINDOW = 14
TRANSITION_Z = 1.5
MIN_OBS = 50
MAX_DATASETS = 15


def load_openesm(max_ds=15, min_obs=50):
    """Load affect data from openESM database."""
    try:
        from openesm import list_datasets, get_dataset
    except ImportError:
        print('openesm not installed. Run: pip install openesm')
        return None
    datasets = list_datasets()
    print(f'Available: {len(datasets)} datasets')
    all_p = []
    loaded = 0
    for ds in datasets:
        if loaded >= max_ds:
            break
        try:
            data = get_dataset(ds)
            if data is None:
                continue
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            akw = ['affect', 'mood', 'happy', 'sad', 'anxious', 'angry',
                   'cheerful', 'relaxed', 'stress', 'depress', 'positive',
                   'negative', 'valence', 'arousal', 'emotion', 'pleasant',
                   'irritable', 'nervous', 'calm', 'content', 'worry']
            acols = [c for c in df.columns if any(k in c.lower() for k in akw)]
            if not acols:
                continue
            pcol = None
            for c in df.columns:
                if any(k in c.lower() for k in
                       ['person', 'participant', 'subject', 'id', 'user', 'ppnr']):
                    pcol = c
                    break
            if pcol is None:
                if df.iloc[:, 0].nunique() < len(df) / 2:
                    pcol = df.columns[0]
                else:
                    continue
            loaded += 1
            dn = getattr(ds, 'name', getattr(ds, 'dataset_id', f'ds_{loaded}'))
            print(f'  [{loaded}] {dn}: {len(df)} obs, '
                  f'{df[pcol].nunique()} persons')
            for pid, g in df.groupby(pcol):
                g = g.reset_index(drop=True)
                if len(g) < min_obs:
                    continue
                num = g[acols].apply(pd.to_numeric, errors='coerce')
                comp = num.mean(axis=1)
                if comp.isna().sum() > len(comp) * 0.5:
                    continue
                all_p.append(pd.DataFrame({
                    'person_id': f'{dn}_{pid}',
                    'time': range(len(g)),
                    'affect': comp.ffill().bfill().values,
                    'dataset': str(dn),
                }))
        except Exception:
            continue
    if all_p:
        r = pd.concat(all_p, ignore_index=True)
        print(f'\nLoaded: {r["person_id"].nunique()} persons, '
              f'{len(r)} obs from {loaded} datasets')
        return r
    return None


def compute_rolling(df, w=14):
    """Compute rolling variance, autocorrelation, entropy."""
    res = []
    for pid, g in df.groupby('person_id'):
        g = g.sort_values('time').reset_index(drop=True)
        a = g['affect'].values
        n = len(a)
        if n < w * 3:
            continue
        for i in range(w, n):
            s = a[i-w:i]
            v = np.var(s)
            m = np.mean(s)
            ac = 0
            if np.std(s) > 1e-6:
                c = np.corrcoef(s[:-1], s[1:])[0, 1]
                ac = c if np.isfinite(c) else 0
            bins = np.linspace(np.min(a) - 0.01, np.max(a) + 0.01, 8)
            cnt = np.histogram(s, bins=bins)[0]
            p = cnt / max(cnt.sum(), 1) + 1e-10
            p = p / p.sum()
            ent = -np.sum(p * np.log(p))
            res.append({
                'person_id': pid, 'time': g.iloc[i]['time'],
                'affect': a[i], 'rolling_var': v, 'rolling_mean': m,
                'rolling_ac': ac, 'rolling_entropy': ent,
            })
    return pd.DataFrame(res)


def find_transitions(met, z=1.5, md=7):
    """Find sustained shifts in rolling mean."""
    tr = []
    for pid, g in met.groupby('person_id'):
        g = g.sort_values('time').reset_index(drop=True)
        ms = g['rolling_mean'].values
        n = len(ms)
        if n < md * 3:
            continue
        gm, gs = np.mean(ms), max(np.std(ms), 1e-6)
        zs = (ms - gm) / gs
        ext = np.abs(zs) > z
        i = 0
        while i < n:
            if ext[i]:
                j = i
                while j < n and ext[j]:
                    j += 1
                if j - i >= md:
                    tr.append({
                        'person_id': pid, 'idx': i,
                        'duration': j - i,
                        'direction': 'pos' if zs[i] > 0 else 'neg',
                        'magnitude': float(np.mean(zs[i:j])),
                    })
                i = j
            else:
                i += 1
    return pd.DataFrame(tr)


def test_pre_transition(met, tr, pw=14):
    """Test pre-transition variance and autocorrelation trends."""
    res = []
    for _, t in tr.iterrows():
        pid, idx = t['person_id'], t['idx']
        pd_ = met[met['person_id'] == pid].sort_values('time').reset_index(drop=True)
        if idx < pw + 2:
            continue
        pre = pd_.iloc[idx - pw:idx]
        if len(pre) < pw // 2:
            continue
        ti = np.arange(len(pre))
        sv, _, _, pv, _ = stats.linregress(ti, pre['rolling_var'].values)
        sa, _, _, pa, _ = stats.linregress(ti, pre['rolling_ac'].values)
        se, _, _, pe, _ = stats.linregress(ti, pre['rolling_entropy'].values)
        res.append({
            'person_id': pid, 'idx': idx, 'direction': t['direction'],
            'var_slope': sv, 'var_p': pv,
            'ac_slope': sa, 'ac_p': pa,
            'ent_slope': se, 'ent_p': pe,
        })
    return pd.DataFrame(res)


def classify_transitions(met, tr, pw=14):
    """Classify transitions as polarizing vs reorganizing."""
    en = []
    for i, t in tr.iterrows():
        pid = t['person_id']
        idx = int(t['idx'])
        pd_ = met[met['person_id'] == pid].sort_values('time').reset_index(drop=True)
        if len(pd_) < idx + pw:
            continue
        gm = pd_['rolling_mean'].mean()
        gs = max(pd_['rolling_mean'].std(), 1e-6)
        gv = pd_['rolling_var'].mean()
        ps = max(0, idx - pw)
        pre = pd_.iloc[ps:idx]
        post = pd_.iloc[idx:min(len(pd_), idx + pw)]
        if len(pre) < 5 or len(post) < 5:
            continue
        pm = pre['rolling_mean'].mean()
        pom = post['rolling_mean'].mean()
        pv = pre['rolling_var'].mean()
        pov = post['rolling_var'].mean()
        prd = abs(pm - gm)
        pod = abs(pom - gm)
        dt_ = 'POLARIZING' if pod > prd else 'REORGANIZING'
        pz = abs(pm - gm) / gs
        pt = 'FROM_EXTREME' if pz > 0.8 else 'FROM_CENTER'
        vl = 'LOW_VAR' if pv < gv * 0.7 else (
            'HIGH_VAR' if pv > gv * 1.3 else 'NORMAL_VAR')
        r = t.to_dict()
        r.update({
            'direction_type': dt_, 'position_type': pt, 'var_level': vl,
            'pre_var': pv, 'post_var': pov,
            'var_change_post': pov - pv,
        })
        en.append(r)
    return pd.DataFrame(en)


def report(test_results, enriched):
    """Print main results."""
    n = len(test_results)
    nd = sum(test_results['var_slope'] < 0)
    ni = n - nd
    na = sum(test_results['ac_slope'] > 0)

    print('=' * 70)
    print('RESULTS: OBSERVABILITY COLLAPSE vs CRITICAL SLOWING')
    print('=' * 70)
    print(f'\n  Transitions analyzed: {n}')
    print(f'  Var down (Obs. Collapse): {nd}/{n} ({nd/n*100:.1f}%)')
    print(f'  Var up (Crit. Slowing):   {ni}/{n} ({ni/n*100:.1f}%)')
    if n >= 5:
        bp = stats.binomtest(nd, n, 0.5)
        print(f'  Binomial p = {bp.pvalue:.4f}')
    print(f'  Autocorr up: {na}/{n} ({na/n*100:.1f}%)')

    if len(enriched) > 0:
        ns = sum(enriched['var_change_post'] > 0)
        fp = sum((enriched['var_slope'] < 0) & (enriched['var_change_post'] > 0))
        cb = sum((enriched['var_slope'] < 0) & (enriched['ac_slope'] > 0))
        print(f'\n  Post-transition spike: {ns}/{len(enriched)} '
              f'({ns/len(enriched)*100:.1f}%)')
        print(f'  Full pattern (silence then spike): {fp}/{len(enriched)} '
              f'({fp/len(enriched)*100:.1f}%) [chance=25%]')
        if len(enriched) >= 5:
            print(f'    Binomial p = '
                  f'{stats.binomtest(fp, len(enriched), 0.25).pvalue:.4f}')
        print(f'  Combined (Var down + AC up): {cb}/{len(enriched)} '
              f'({cb/len(enriched)*100:.1f}%) [chance=25%]')

    # Split test
    print(f'\n  SPLIT BY TRANSITION TYPE:')
    for dt_ in ['POLARIZING', 'REORGANIZING']:
        sub = enriched[enriched['direction_type'] == dt_]
        if len(sub) >= 3:
            nd_ = sum(sub['var_slope'] < 0)
            print(f'    {dt_}: Var down in {nd_}/{len(sub)} '
                  f'({nd_/len(sub)*100:.1f}%)')


if __name__ == '__main__':
    print('OBSERVABILITY COLLAPSE — EMPIRICAL ANALYSIS')
    print('=' * 70)

    df = load_openesm(MAX_DATASETS, MIN_OBS)
    if df is None:
        print('\nCould not load openESM data. Exiting.')
        exit(1)

    print(f'\nData: {df["person_id"].nunique()} persons, {len(df)} obs')

    print('\nComputing rolling metrics...')
    metrics = compute_rolling(df, WINDOW)
    print(f'  {len(metrics)} windowed obs, {metrics["person_id"].nunique()} persons')

    print('\nDetecting transitions...')
    trans = find_transitions(metrics, TRANSITION_Z)
    print(f'  {len(trans)} transitions in '
          f'{trans["person_id"].nunique() if len(trans) > 0 else 0} persons')

    if len(trans) == 0:
        print('\nNo transitions detected. Adjust thresholds.')
        exit(0)

    print('\nTesting pre-transition patterns...')
    test_results = test_pre_transition(metrics, trans, WINDOW)
    print(f'  {len(test_results)} transitions with sufficient pre-data')

    print('\nClassifying transitions...')
    enriched = classify_transitions(metrics, test_results, WINDOW)
    print(f'  {len(enriched)} classified')

    if len(test_results) > 0:
        report(test_results, enriched)

    # Save results
    if len(enriched) > 0:
        enriched.to_csv('empirical_results.csv', index=False)
        print('\nResults saved to empirical_results.csv')
