from src.analysis.paper_exports import populate_main_tex


def test_populate_main(tmp_path):
    p = tmp_path / 'main.tex'
    p.write_text('\\section{Results}\n% Intentionally left blank per request. Populate with curves/tables above.')
    populate_main_tex(str(p))
    assert 'main_results' in p.read_text()
