# PICR-MA

This repository now contains a **resource-safe, reproducible scaffold** for PICR-MA experiments.

## What PICR-MA is
PICR-MA is a consistency regularizer for MARL that enforces permutation invariance over teammate feature blocks by minimizing a symmetric KL divergence between policy outputs for two independently permuted teammate orders.

## Current scope
- Active fallback profile support via preflight.
- Matrix-game pipeline implemented and runnable without heavy dependencies.
- MPE hooks and manuscript population pipeline scaffolded for later full runs once torch/pettingzoo are installed.

## Quickstart
```bash
make preflight
make smoke
make stage1
make plots
make tables
make paper
make report
```

## Safety
- Uses profile-based budgets and artifact inventory.
- Stops at scaffold-only when core dependencies are unavailable.

## Limitations
- Full IPPO/PettingZoo experiments require torch + pettingzoo + supersuit installations.
- Current runs are matrix-game-only under scaffold profile.
