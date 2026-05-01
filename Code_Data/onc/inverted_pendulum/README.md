## Inverted Pendulum

This folder contains a Gymnasium `InvertedPendulum-v5` ONC runner with:

- single-run optimize/replay support
- `experiment_full` for full 4D observation input
- `experiment_partial` for partial 2D observation input with velocity removed
- automatic multi-seed plot and final-return summary output

### Commands

Full-observation experiment:

```bash
python Code_Data/inverted_pendulum/inverted_pendulum.py --experiment_full
```

Partial-observation experiment:

```bash
python Code_Data/inverted_pendulum/inverted_pendulum.py --experiment_partial
```

Single-run optimize:

```bash
python Code_Data/inverted_pendulum/inverted_pendulum.py --optimize
```
