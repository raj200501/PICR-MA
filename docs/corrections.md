# Corrections to earlier versions

Versions of this repository up to commit `48235c1` (April 2026), including the
workshop-paper draft built from them, contained implementation errors. **Every
result produced by that code is invalid.** The code was rewritten, and all
experiments were re-run from scratch with the corrected implementation. The old
result files and paper builds were removed; they remain in git history.

| Defect in the earlier code | Effect | Fix |
|---|---|---|
| The PPO policy gradient had the wrong sign: `dlogits` used `(probs - one_hot)` where the loss gradient is `(one_hot - probs)`. | Each update moved the policy *away* from high-advantage actions. On a one-step bandit, the rewarded action's probability went to zero. On `simple_spread_v3`, training return got worse, not better. | Learner rewritten in PyTorch with autodiff; `tests/test_ppo.py::test_ppo_learns_contextual_bandit` fails if the sign is flipped back. |
| GAE ran over a buffer that interleaved agents (agent 0, 1, 2, agent 0, …) as if it were one trajectory. | Each agent bootstrapped from a *teammate's* value estimate. | GAE now runs per agent along time; checked against a reference implementation in `tests/test_ppo.py`. |
| Cross-play loaded a checkpoint selected by the return of one stochastic training episode, while every other metric used the final weights. | Cross-play compared different policies from the other metrics, and the choice of policy was noisy. | All evaluation uses the final policy. |
| Training environment seeds were `seed + episode`, and evaluation used seeds from the same range. | Training runs with different seeds shared episodes, and evaluation states overlapped training states. | Training and evaluation seeds come from disjoint ranges; each run has an independent random stream. |
| Different methods with the same seed shared initial weights and training episodes. | "Independently trained" partners from different methods were near-clones. | Seed streams depend on (seed, method, λ). |
| In `simple_adversary_v3`, each team was scored against its own co-trained adversary. | Methods were compared against different opponents. | All cross-play and self-play evaluation uses one shared pool of adversaries. |
| Permutation sensitivity was measured on the first 64 states of training episode 0 (the untrained policy) and included the identity permutation. | The reported "invariance" numbers did not describe the trained policies. | Measured on evaluation states visited by the final policy, over non-identity permutations only. |
| The headline experiments had no launch script, and requirements were unpinned (current PettingZoo no longer ships MPE). | The paper's numbers could not be regenerated. | Every run is defined in `configs/experiments.yaml`; dependencies are pinned; `make reproduce` runs the whole pipeline. |
| The "Other-Play" baseline only permuted the order of teammate inputs. | The label overstated it. | Renamed *order augmentation*; the paper describes exactly what it does. |

The earlier draft's conclusions, including the "invariance–coordination mismatch",
were drawn from these invalid results and should not be cited.
