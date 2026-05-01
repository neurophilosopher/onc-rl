## Swimmer

This folder contains a Gymnasium `Swimmer-v5` ONC runner with:

- single-run optimize/replay support
- `experiment_full` for full 8D observation input
- `experiment_partial` for partial 2D observation input with velocity removed
- automatic multi-seed plot and final-return summary output

### Commands

Full-observation experiment:

```bash
python Code_Data/swimmer/swimmer.py --experiment_full
```

Partial-observation experiment:

```bash
python Code_Data/swimmer/swimmer.py --experiment_partial
```

Single-run optimize:

```bash
python Code_Data/swimmer/swimmer.py --optimize
```
