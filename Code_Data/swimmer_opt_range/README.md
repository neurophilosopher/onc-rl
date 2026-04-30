## Swimmer With ARS-Optimized Sensory Range

This folder contains a Gymnasium `Swimmer-v5` ONC runner with:

- single-run optimize/replay support
- `experiment_full` for full 8D observation input
- `experiment_partial` for partial 2D observation input with velocity removed
- automatic multi-seed plot and final-return summary output
- a single trainable symmetric sensory-range scale optimized by ARS

The bi-sensory encoder in this variant uses a fixed interface range of `[-1, 1]`,
while the projected ONC inputs are divided by a learned positive scalar
`sensory_range_scale`. This is equivalent to optimizing a shared symmetric
bi-sensory range `[-R, R]` for both input channels.

### Commands

Full-observation experiment:

```bash
python Code_Data/swimmer_opt_range/swimmer.py --experiment_full
```

Partial-observation experiment:

```bash
python Code_Data/swimmer_opt_range/swimmer.py --experiment_partial
```

Single-run optimize:

```bash
python Code_Data/swimmer_opt_range/swimmer.py --optimize
```
