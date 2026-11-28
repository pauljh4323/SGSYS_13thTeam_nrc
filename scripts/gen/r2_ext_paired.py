"""exp/r2-ext: paired PESQ deltas of gru_v8p2z6a/b vs gru_v8p2z3e from eval.py per-file CSVs (outputs/<cfg>/eval_<split>.csv).
Run from the repo root: python scripts/gen/r2_ext_paired.py > logs/r2_ext_paired.txt"""
import sys, numpy as np, pandas as pd
O='outputs'
base='gru_v8p2z3e'; models=['gru_v8p2z6a','gru_v8p2z6b']
def load(m,f): return pd.read_csv(f'{O}/{m}/{f}').set_index('segment_id')
def corp(code):
    parts=code.split('+'); c=sorted({('DEMAND' if p.startswith('demand') else 'MUSAN' if p.startswith('musan') else 'ESC-50') for p in parts}); return '+'.join(c)
def pd_(a,b):
    d=(a.pesq-b.pesq).dropna(); se=d.std(ddof=1)/np.sqrt(len(d)); return d.mean(),se,d.mean()/se if se>0 else np.nan,len(d)
def fmt(t): return f'{t[0]:+.4f} ± {t[1]:.4f} (z {t[2]:+.1f}, n {t[3]})'
sets={'validation':'eval_validation.csv','val_remix':'eval_val_remix.csv','gen_unseen_test':'eval_gen_unseen_test.csv',
      'gen_multi_test':'eval_gen_multi_test.csv','gen_low':'eval_gen_low.csv'}
for m in models:
    print(f'=== {m}')
    pool=[]
    for s,f in sets.items():
        try: a=load(m,f); b=load(base,f)
        except FileNotFoundError as e: print(s,'missing'); continue
        b=b.loc[a.index]
        print(f'{s:16s} {m} {a.pesq.mean():.4f}/{a.snr_db.mean():.2f} dB  z3e {b.pesq.mean():.4f}/{b.snr_db.mean():.2f}  delta {fmt(pd_(a,b))}  fail {a.pesq.isna().sum()}')
        print('   per SNR', {k:f'{g.pesq.mean():.3f} ({fmt(pd_(g,b.loc[g.index]))})' for k,g in a.groupby('target_snr_db')})
        if s.startswith('gen'):
            pool.append((a.pesq-b.pesq).rename(s))
            if s!='gen_low':
                man=pd.read_csv(f'data/gen/{s}/samples.csv').set_index('segment_id'); a=a.assign(c=man.loc[a.index,'noise_collection'].map(corp))
                for k,g in a.groupby('c'): print(f'   {k:14s} {g.pesq.mean():.3f} z3e {b.loc[g.index].pesq.mean():.3f} {fmt(pd_(g,b.loc[g.index]))}')
        if s=='validation':
            a=a.assign(spk=a.index.str.split('_').str[0])
            r=[(k,len(g),pd_(g,b.loc[g.index])[0]) for k,g in a.groupby('spk')]
            ds=np.array([x[2] for x in r]); print(f'   speakers {len(r)}: delta>0 {int((ds>0).sum())}, min {ds.min():+.3f} ({r[ds.argmin()][0]}), max {ds.max():+.3f} ({r[ds.argmax()][0]})')
            for k,g in a.groupby('noise_difficulty'): print(f'   diff {k}: {g.pesq.mean():.3f} {fmt(pd_(g,b.loc[g.index]))}')
    if len(pool)==3:
        d=pd.concat(pool).dropna(); se=d.std(ddof=1)/np.sqrt(len(d)); print(f'POOLED gen n {len(d)} delta {d.mean():+.4f} ± {se:.4f} z {d.mean()/se:+.2f}')
    pr=[]
    for g in (-30,-20,-10,0):
        try: pr.append((g,pd.read_csv(f'{O}/{m}/eval_validation_gain{g}dB.csv').pesq.mean()))
        except FileNotFoundError: pass
    if pr: v=[x[1] for x in pr]; print('probe', [(g,round(x,4)) for g,x in pr], 'range', round(max(v)-min(v),4))
if len(models)==2:
    for s,f in sets.items():
        try: a=load(models[1],f); b=load(models[0],f).loc[a.index]; print(f'z6b-z6a {s}: {fmt(pd_(a,b))}')
        except FileNotFoundError: pass
for f in ('eval_gen_unseen_dev.csv','eval_gen_multi_dev.csv'):
    try: print(base,f,round(load(base,f).pesq.mean(),4))
    except FileNotFoundError: pass
