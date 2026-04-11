# Main Blocked

Main is blocked for this branch.

## Blocking reasons

1. `PICR-MA` does not clear the method-claim bar.
   - It improves held-out cross-play on `simple_adversary_v3`.
   - It underperforms IPPO, Deep Sets, and Other-Play on `simple_spread_v3`.
   - This is a mixed tradeoff story, not a robust new default.

2. The strongest baseline story is still unfavorable.
   - Other-Play is slightly better than PICR-MA on adversary cross-play and clearly better on spread.
   - Deep Sets remains competitive on spread and dominates the invariance diagnostic.

3. The mechanistic claim is weak.
   - Lower permutation sensitivity does not reliably track better zero-shot coordination.
   - The best 20k cross-play lambda is not the most invariant lambda.

4. The empirical scale is improved but still modest.
   - Main MPE runs are 50k steps with 5 seeds.
   - `simple_spread_v3` curves are still improving late in training.
   - The project does not yet look fully out of toy-scale territory.

5. Implementation credibility remains limited.
   - The artifact still uses a custom NumPy learner rather than a standard PyTorch PPO stack.

## What would need to change before main approval

- A stronger main-cohort result that survives both MPE environments.
- Either a clearer win over IPPO/Other-Play or a more convincing tradeoff claim against Deep Sets and Other-Play.
- Stronger support for the invariance-to-ZSC mechanism.
- Larger and more settled training runs, or a stronger argument that the current scale is sufficient.
- Preferably, a more standard learner backend.
