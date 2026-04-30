## pybnn4

This is a parallel copy of `Code_Data/pybnn` intended for experiments that use a
separate Python extension module name, `pybnn4`.

The core `LifNet` implementation already supports an arbitrary number of sensory
channels through repeated calls to:

- `AddSensoryNeuron(...)`
- `AddBiSensoryNeuron(...)`

So this folder does not hardcode "4 inputs" in the C++ core. Instead, it
provides an isolated module name and build target that can be used by a new
experiment script wiring 4 sensory inputs without modifying the existing
`pybnn`-based experiments.

### Build

From the repo root:

```bash
cd Code_Data/pybnn4
make
```

This should produce a Python extension named `pybnn4...so` in `bin/`.
