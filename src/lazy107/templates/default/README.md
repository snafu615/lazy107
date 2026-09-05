# {project}

A training project scaffolded with `lazy107`.

## Usage

1. Set up the environment: `lazy107 env --yes`
2. Implement `src/data.py` and `src/model.py`
3. Review the resolved plan: `lazy107 plan`
4. Submit to Slurm: `lazy107 submit --yes`

## Layout

- `src/data.py` - dataset loading and preprocessing
- `src/model.py` - model architecture
- `src/train.py` - training loop
- `scripts/train.sbatch` - generated Slurm batch script
- `107.toml` - optional project overrides

## Tracking runs

Each submission appends an entry to `notes/runs.md`; job output lands in
`logs/<job-name>_<job-id>.{out,err}`.
