# Bio-Inspired Neural Networks for Reinforcement Learning

LSTM controller experiments for the COMP579 project, reproducing parts of
Hasani et al. (2020) on MuJoCo control tasks (InvertedPendulum, HalfCheetah,
Swimmer) with both ARS and PPO.

## Setup

Create a virtual environment and install dependencies:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the experiments

All sweeps are launched from `main.py`. Each mode runs 5 seeds across the
relevant environments in parallel and saves per-seed results as JSON in
`runs/`, then generates plots.

```
python main.py ars              # ARS, hidden_dim=12, no projection
python main.py ppo              # PPO, hidden_dim=12, no projection
python main.py all              # ARS and PPO back-to-back
python main.py ars_proj         # ARS, hidden_dim=12, with 2D input projection
python main.py ppo_small_proj   # PPO, hidden_dim=12, with 2D input projection
python main.py ppo_big          # PPO, hidden_dim=64, no projection
python main.py extras           # ppo_small_proj, ppo_big, ars_proj
```

Plots are written to `plots/`, `plots_proj2/`, or `plots_h64/` depending on
the mode.

## File layout

- `main.py` — parallel sweep launcher (CLI entry point)
- `jobs.py` — single-experiment worker
- `training.py` — ARS and PPO training loops with periodic evaluation
- `policies.py` — LSTM policy architectures (vanilla and projected variants)
- `ppo_utils.py` — observation normalizer, rollout collection, GAE, PPO update
- `evaluation.py` — deterministic evaluation rollouts
- `envs.py` — environment wrappers (partial observability, reward shaping)
- `plotting.py` — aggregates per-seed JSON results into learning curves