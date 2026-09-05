# lazy107 Documentation

lazy107 is a cluster-side Slurm submission CLI for the USTC 107 platform.
Start with the [Quick Start](quickstart.md) if you are new; dive into
[Design](design.md) if you want to understand how it works.

## Document index

| Doc | Audience | Contents |
|---|---|---|
| [quickstart.md](quickstart.md) | First-time users | 5 minutes from install to first submitted job |
| [design.md](design.md) | Developers, maintainers | Architecture, CLI structure, config system, GPU/CUDA integrity, safety boundaries |
| [env-pipeline.md](env-pipeline.md) | Developers, maintainers | The env pipeline end to end: analysis → generation → execution → consumption |
| [demo.md](demo.md) | Anyone curious | End-to-end worked example (scaffold → env → render → submit) |
| [runbook.md](runbook.md) | Acceptance testing | Detailed steps, acceptance criteria, common blockers |

## Language note

All documents are in English. The platform facts (partitions, QoS, modules,
mirrors) are cross-checked against the USTC 107 training deck and the
Slurm script contract for the platform.
