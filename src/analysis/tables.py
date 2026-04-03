from pathlib import Path
import csv

def _write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)

def make_tables(rows, outdir='latex/tables'):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    names=['main_results.tex','crossplay_results.tex','permutation_sensitivity.tex','runtime_results.tex']
    if not rows:
        for n in names: _write(Path(outdir)/n, '% No data available\n')
        return
    lines=['\\begin{tabular}{llccc}','\toprule','Env & Method & Train & Cross-play & JS \\','\midrule']
    for r in rows:
        lines.append(f"{r['env']} & {r['method']} & {r['train_return']:.3f} & {r['crossplay_return']:.3f} & {r['permutation_js']:.3f} \\")
    lines+=['\bottomrule','\end{tabular}','']
    txt='\n'.join(lines)
    for n in names: _write(Path(outdir)/n,txt)
