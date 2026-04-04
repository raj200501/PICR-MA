from __future__ import annotations

import re
from pathlib import Path


RESULTS_BLOCK = r"""
\section{Results}
\label{sec:results}
\paragraph{Main performance and zero-shot coordination.}
Table~\ref{tab:main-results} summarizes the currently available runs. Figure~\ref{fig:learning-curves} shows the training curves on the MPE tasks that completed under the active profile, while Figure~\ref{fig:zsc-heatmaps} visualizes the held-out partner cross-play matrices generated from saved checkpoints.

\begin{table*}[t]
\centering
\caption{Main performance summary from grounded run artifacts. Lower permutation sensitivity is better. Cross-play entries remain blank when a held-out partner pool was not yet available for that environment/method.}
\label{tab:main-results}
\input{latex/tables/main_results.tex}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\linewidth]{latex/figures/learning_curves_mpe.pdf}
\caption{MPE learning curves from the saved runs available in \texttt{results/}. Shaded regions show standard error when multiple seeds are available.}
\label{fig:learning-curves}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\linewidth]{latex/figures/zero_shot_crossplay_results.pdf}
\caption{Held-out partner cross-play heatmaps aggregated from checkpoint-based evaluations.}
\label{fig:zsc-heatmaps}
\end{figure*}

\paragraph{Permutation sensitivity.}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{latex/figures/permutation_sensitivity_results.pdf}
\caption{Policy sensitivity to teammate-order permutations, measured with Jensen--Shannon divergence over action distributions.}
\label{fig:perm-sensitivity}
\end{figure}

\paragraph{Compute and matrix-game diagnostics.}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{latex/figures/runtime_overhead.pdf}
\caption{Observed wall-clock runtime across the available runs.}
\label{fig:runtime-overhead}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{latex/figures/optional_matrix_games_summary.pdf}
\caption{Matrix-game summary under the current small-compute regime.}
\label{fig:matrix-summary}
\end{figure}
"""


def populate_main_tex(main_tex: str | Path = "main.tex") -> None:
    path = Path(main_tex)
    text = path.read_text(encoding="utf-8")
    text = text.replace("\\section{Experimental Setup (No Results Yet)}", "\\section{Experimental Setup}")
    text = re.sub(
        r"% --- Placeholders.*?\\section\{Results\}.*?\\section\{Discussion and Limitations \(to accompany Results\)\}",
        lambda _: RESULTS_BLOCK + "\n\\section{Discussion and Limitations (to accompany Results)}",
        text,
        flags=re.S,
    )
    if "method_overview.pdf" not in text:
        insert_after = "\\paragraph{Why this matters.} In many cooperative domains (robot swarms, sensor networks, traffic), teammate ordering in observations is semantically meaningless. Yet vanilla concatenation forces a spurious ordering that policies exploit. Enforcing invariance via a lightweight regularizer is a practical alternative to bespoke set-/graph-equivariant architectures \\citep{zaheer2017deepsets, liu2020pic}, immediately useful where simplicity and limited compute are priorities.\n"
        overview = (
            insert_after
            + "\n\\begin{figure*}[t]\n\\centering\n"
            + "\\includegraphics[width=0.92\\linewidth]{latex/figures/method_overview.pdf}\n"
            + "\\caption{PICR-MA overview. The policy sees two independently permuted teammate-order views of the same observation, and training penalizes disagreement between the resulting action distributions.}\n"
            + "\\label{fig:method-overview}\n\\end{figure*}\n"
        )
        text = text.replace(insert_after, overview)
    path.write_text(text, encoding="utf-8")
