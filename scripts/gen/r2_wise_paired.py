"""exp/r2-ext WiSE-FT: paired PESQ deltas of gru_v8p2zw<alpha> vs gru_v8p2z3e from eval.py per-file CSVs.
Run from the repo root:
  python scripts/gen/r2_wise_paired.py dev            > logs/r2_wise_dev.txt    (alpha sweep + fixed selection rule)
  python scripts/gen/r2_wise_paired.py test <alpha>   > logs/r2_wise_test.txt   (chosen alpha only)
Selection rule (fixed before any dev number was seen): among alphas with validation paired delta >= -0.02, the
highest mean of gen_unseen_dev and gen_multi_dev PESQ; if none qualifies, the alpha with the smallest validation loss."""
import sys, numpy as np, pandas as pd
O = 'outputs'; base = 'gru_v8p2z3e'; ALPHAS = [20, 30, 40, 50, 60, 70, 80]
def load(m, s): return pd.read_csv(f'{O}/{m}/eval_{s}.csv').set_index('segment_id')
def pdelta(a, b):
    d = (a.pesq - b.loc[a.index].pesq).dropna(); se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se if se > 0 else np.nan, len(d)
def fmt(t): return f'{t[0]:+.4f} ± {t[1]:.4f} (z {t[2]:+.1f}, n {t[3]})'

def dev():
    sets = ['validation', 'gen_unseen_dev', 'gen_multi_dev']
    b = {s: load(base, s) for s in sets}
    print('alpha | ' + ' | '.join(f'{s} (delta)' for s in sets) + ' | dev mean | qualifies')
    print(f'0.0 (z3e) | ' + ' | '.join(f'{b[s].pesq.mean():.4f}' for s in sets) +
          f' | {(b["gen_unseen_dev"].pesq.mean() + b["gen_multi_dev"].pesq.mean()) / 2:.4f} | yes')
    rows = []
    for a in ALPHAS:
        m = f'gru_v8p2zw{a}'; r = {s: load(m, s) for s in sets}
        d = {s: pdelta(r[s], b[s]) for s in sets}
        devm = (r['gen_unseen_dev'].pesq.mean() + r['gen_multi_dev'].pesq.mean()) / 2
        q = d['validation'][0] >= -0.02
        rows.append((a, d['validation'][0], devm, q))
        print(f'0.{a // 10} | ' + ' | '.join(f'{r[s].pesq.mean():.4f} / {r[s].snr_db.mean():.2f} dB ({fmt(d[s])})' for s in sets)
              + f' | {devm:.4f} | {"yes" if q else "no"}')
    ok = [x for x in rows if x[3]]
    pick = max(ok, key=lambda x: x[2]) if ok else max(rows, key=lambda x: x[1])
    print(f'CHOSEN alpha 0.{pick[0] // 10} ({"rule: max dev mean among val delta >= -0.02" if ok else "fallback: smallest val loss"})'
          f' val delta {pick[1]:+.4f} dev mean {pick[2]:.4f}')

def test(a):
    m = f'gru_v8p2zw{a}'
    sets = ['validation', 'val_remix', 'gen_unseen_test', 'gen_multi_test', 'gen_low']
    pool = []
    for s in sets:
        r, b = load(m, s), load(base, s)
        print(f'{s:16s} {m} {r.pesq.mean():.4f}/{r.snr_db.mean():.2f} dB  z3e {b.loc[r.index].pesq.mean():.4f}/'
              f'{b.loc[r.index].snr_db.mean():.2f}  delta {fmt(pdelta(r, b))}  fail {r.pesq.isna().sum()}')
        print('   per SNR', {k: f'{g.pesq.mean():.3f} ({fmt(pdelta(g, b))})' for k, g in r.groupby('target_snr_db')})
        if s.startswith('gen'): pool.append(r.pesq - b.loc[r.index].pesq)
    d = pd.concat(pool).dropna(); se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'POOLED gen n {len(d)} delta {d.mean():+.4f} ± {se:.4f} z {d.mean() / se:+.2f}')
    pr = [(g, pd.read_csv(f'{O}/{m}/eval_validation_gain{g}dB.csv').pesq.mean()) for g in (-30, -20, -10, 0)]
    v = [x[1] for x in pr]; print('probe', [(g, round(x, 4)) for g, x in pr], 'range', round(max(v) - min(v), 4))

if __name__ == '__main__':
    dev() if sys.argv[1] == 'dev' else test(int(sys.argv[2]))
