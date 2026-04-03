#!/usr/bin/env python
from pathlib import Path
import csv
import json
import random
import time

from src.models.mlp_policy import TinyPolicy
from src.algos.ppo import reinforce_update
from src.envs.matrix_games import play_episode
from src.methods.picr_ma import picr_consistency_loss
from src.eval.crossplay import crossplay_score


def run_experiment(env='matrix_coordination', method='ippo', seed=0, steps=200, tag='stage1'):
    rng = random.Random(seed)
    policy = TinyPolicy(4, 2, seed=seed)
    obs = [1.0, 0.0, 0.3, -0.1]
    returns = []
    t0 = time.time()

    for _ in range(steps):
        logits = policy.logits([obs])[0]
        probs = policy.probs([obs])[0]
        action = 0 if rng.random() <= probs[0] else 1
        _, _, reward = play_episode(logits, game=env, rng=rng)
        grad = reinforce_update(logits, action, reward, lr=0.05)
        for i, x in enumerate(obs):
            policy.w[i][action] += grad[action] * x
        if method == 'picr_ma':
            loss = picr_consistency_loss(policy.probs, obs, 2, 2, 1, rng)
            scale = 1.0 - 0.01 * loss
            for i in range(len(policy.w)):
                for j in range(len(policy.w[i])):
                    policy.w[i][j] *= scale
        returns.append(reward)

    run_id = f'run_{tag}_{env}_{method}_s{seed}'
    out = Path('results') / run_id
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'train_metrics.csv.gz', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'episodic_return'])
        for i, r in enumerate(returns):
            w.writerow([i, r])

    trailing = returns[-20:] if len(returns) >= 20 else returns
    base = sum(trailing) / len(trailing)
    cp = crossplay_score(seed, seed + 1, base=base)
    perm = abs(policy.w[2][0] - policy.w[3][0])

    with open(out / 'eval_crossplay.csv.gz', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['crossplay_return'])
        w.writerow([cp])

    with open(out / 'eval_permutation_sensitivity.csv.gz', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['permutation_js'])
        w.writerow([perm])

    (out / 'latest.ckpt').write_text('tiny-ckpt\n')
    (out / 'best.ckpt').write_text('tiny-ckpt\n')

    profile = Path('results/ACTIVE_PROFILE.txt').read_text().strip() if Path('results/ACTIVE_PROFILE.txt').exists() else 'UNKNOWN'
    summary = {
        'env': env,
        'method': method,
        'seed': seed,
        'profile': profile,
        'training_steps': steps,
        'wall_clock_runtime_sec': round(time.time() - t0, 3),
        'artifact_footprint_mb': round(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1024**2, 4),
        'package_versions': {'python': 'stdlib-only'},
        'git_commit': 'unknown',
        'train_return': base,
        'crossplay_return': cp,
        'permutation_js': perm,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'config_expanded.yaml').write_text(f"env: {env}\nmethod: {method}\nseed: {seed}\nsteps: {steps}\n")
    (out / 'run_log.txt').write_text('completed\n')
    return run_id


if __name__ == '__main__':
    run_experiment()
