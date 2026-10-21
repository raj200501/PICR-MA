from picr_ma.analysis.paper_exports import populate_main_tex


def test_populate_main(tmp_path):
    main_tex = tmp_path / "main.tex"
    main_tex.write_text(
        "\\section{Experimental Setup (No Results Yet)}\n"
        "% --- Placeholders to help plan the 8-page budget ---\n"
        "\\placeholderfigure{X}{fig:x}\n"
        "\\section{Results}\n"
        "\\input{latex/tables/main_results.tex}\n"
        "\\section{Discussion and Limitations (to accompany Results)}\n",
        encoding="utf-8",
    )
    populate_main_tex(main_tex)
    updated = main_tex.read_text(encoding="utf-8")
    assert "method_overview.pdf" in updated or "learning_curves_mpe.pdf" in updated
