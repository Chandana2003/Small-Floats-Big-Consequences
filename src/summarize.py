import pandas as pd
import numpy as np
import os

def summarize_gemm(path='results/gemm.csv'):
    df = pd.read_csv(path)
    # use fp32 as per-n baseline for each (n, kappa)
    base = df[(df.fmt=='fp32') & (df.accum=='fp32')][['n','kappa','rel_error']]
    base = base.rename(columns={'rel_error':'base_err'})
    out = df.merge(base, on=['n','kappa'], how='left')
    def label(row):
        e = row['rel_error']; b = row['base_err']
        if np.isnan(b): return '—'
        if e <= 1.5*b: return 'Safe'
        if e <= 5e-2:  return 'Borderline'
        return 'Risky'
    out['label'] = out.apply(label, axis=1)
    return out[['n','kappa','fmt','accum','rel_error','base_err','label']]

def summarize_cg(path='results/cg.csv'):
    df = pd.read_csv(path)
    # fp32 iterations as baseline per m
    base = df[(df.fmt=='fp32') & (df.accum=='fp32')][['m','iters']]
    base = base.rename(columns={'iters':'iters_base'})
    out = df.merge(base, on='m', how='left')
    def label(row):
        it = row['iters']; b = row['iters_base']; fr = row['final_resid']
        if np.isnan(b): return '—'
        # if didn't reduce residual “enough”, mark risky
        if fr > 1e-3 and it >= row['iters_base']*1.5: return 'Risky'
        if it <= 1.5*b: return 'Safe'
        return 'Borderline'
    out['label'] = out.apply(label, axis=1)
    return out[['m','fmt','accum','iters','final_resid','iters_base','label']]

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    g = summarize_gemm()
    c = summarize_cg()
    g.to_csv('results/gemm_summary.csv', index=False)
    c.to_csv('results/cg_summary.csv', index=False)
    print('Wrote results/gemm_summary.csv and results/cg_summary.csv')
