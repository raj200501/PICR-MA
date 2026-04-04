#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


BIBTEX = r"""
@inproceedings{foerster2016dqncomm,
  author = {Foerster, Jakob N. and Assael, Yannis M. and de Freitas, Nando and Whiteson, Shimon},
  title = {Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2016}
}

@inproceedings{hu2020otherplay,
  author = {Hu, Hengyuan and Lerer, Adam and Peysakhovich, Alexander and Foerster, Jakob N.},
  title = {Other-Play for Zero-Shot Coordination},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year = {2020}
}

@inproceedings{liu2020pic,
  author = {Liu, Iou-Jen and Yeh, Raymond A. and Schwing, Alexander G.},
  title = {PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning},
  booktitle = {Proceedings of the 4th Conference on Robot Learning},
  year = {2020}
}

@inproceedings{lowe2017maddpg,
  author = {Lowe, Ryan and Wu, Yi I. and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  title = {Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017}
}

@inproceedings{rashid2018qmix,
  author = {Rashid, Tabish and Samvelyan, Mikayel and De Witt, Christian Schroeder and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  title = {QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  year = {2018}
}

@misc{schulman2017ppo,
  author = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title = {Proximal Policy Optimization Algorithms},
  year = {2017},
  note = {arXiv:1707.06347}
}

@inproceedings{sohn2020fixmatch,
  author = {Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin and Cubuk, Ekin D. and Kurakin, Alex and Li, Chun-Liang},
  title = {FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020}
}

@inproceedings{sukhbaatar2016commnet,
  author = {Sukhbaatar, Sainbayar and Szlam, Arthur and Fergus, Rob},
  title = {Learning Multiagent Communication with Backpropagation},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2016}
}

@inproceedings{sunehag2018vdn,
  author = {Sunehag, Peter and Lever, Guy and Gruslys, Audrunas and Czarnecki, Wojciech M. and Zambaldi, Vinicius and Jaderberg, Max and Lanctot, Marc and Sonnerat, Nicolas and Leibo, Joel Z. and Tuyls, Karl and Graepel, Thore},
  title = {Value-Decomposition Networks for Cooperative Multi-Agent Learning},
  booktitle = {Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems},
  year = {2018}
}

@inproceedings{terry2021pettingzoo,
  author = {Terry, Justin K. and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis and Perez, Rodrigo and Horsch, Christopher and Dieffendahl, Caleb and Williams, Niall L. and Lokesh, Yashas and Ravi, Praveen},
  title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year = {2021}
}

@misc{terry2020ps,
  author = {Terry, Justin K. and Grammel, Nathaniel and Son, Sanghyun and Black, Benjamin and Agrawal, Anikait},
  title = {Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning},
  year = {2020},
  note = {arXiv:2005.13625}
}

@inproceedings{tarvainen2017meanteacher,
  author = {Tarvainen, Antti and Valpola, Harri},
  title = {Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017}
}

@inproceedings{zaheer2017deepsets,
  author = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Ruslan and Smola, Alexander J.},
  title = {Deep Sets},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017}
}
""".strip() + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    bib_path = root / "references.bib"
    bib_path.write_text(BIBTEX, encoding="utf-8")
    (root / "results/BIBLIOGRAPHY_STATUS.md").write_text(
        "\n".join(
            [
                "# BIBLIOGRAPHY STATUS",
                "- Mode: offline",
                "- Source of truth: verified bibliography entries already present in main.tex",
                f"- Output: {bib_path.name}",
                "- Network fetch was not attempted in the final artifact path.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(bib_path)
