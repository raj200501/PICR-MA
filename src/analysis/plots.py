from pathlib import Path


def _write_min_pdf(path, title, lines):
    body = " ; ".join([title] + lines).replace('(', '[').replace(')', ']')
    txt = f"BT /F1 12 Tf 50 750 Td ({body}) Tj ET"
    content = f"""%PDF-1.4
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj
4 0 obj<< /Length {len(txt)} >>stream
{txt}
endstream endobj
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000062 00000 n 
0000000120 00000 n 
0000000260 00000 n 
0000000000 00000 n 
trailer<< /Root 1 0 R /Size 6 >>
startxref
0
%%EOF
"""
    Path(path).write_text(content)


def make_basic_plots(rows, outdir='latex/figures'):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if not rows:
        return []

    by_method = {}
    for r in rows:
        by_method.setdefault(r['method'], []).append(r)

    lines1 = [f"{m}: train={sum(x['train_return'] for x in rs)/len(rs):.3f}" for m, rs in by_method.items()]
    lines2 = [f"{m}: cross={sum(x['crossplay_return'] for x in rs)/len(rs):.3f}" for m, rs in by_method.items()]
    lines3 = [f"{m}: js={sum(x['permutation_js'] for x in rs)/len(rs):.3f}" for m, rs in by_method.items()]

    p1 = Path(outdir) / 'learning_curves_mpe.pdf'
    p2 = Path(outdir) / 'zero_shot_crossplay_results.pdf'
    p3 = Path(outdir) / 'permutation_sensitivity_results.pdf'
    _write_min_pdf(p1, 'Learning summary', lines1)
    _write_min_pdf(p2, 'Cross-play summary', lines2)
    _write_min_pdf(p3, 'Sensitivity summary', lines3)
    return [str(p1), str(p2), str(p3)]
