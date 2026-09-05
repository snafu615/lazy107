# 端到端演示：从零到训练好的模型

本文档记录一次完整的真实训练：用 lazy107 把
[`examples/resnet-cifar10`](../examples/resnet-cifar10)（ResNet-18 训练
CIFAR-10）提交到 `P107-RTX5090` 上跑完，拿到真实指标。
每个步骤 = 命令 → 输出 → 这段说明了什么。数值以实际运行为准，
格式与流程一致。

> 更多主题式演示（安装探测、happy path、everything 向导、debug、
> 自适应渲染、防护栏杆）见 [docs/demos/INDEX.md](demos/INDEX.md)。

## 场景

- 项目：`examples/resnet-cifar10`（纯 `.py`，**没有任何依赖文件**——
  刻意展示「只靠代码 import 扫描」的推断路径）
- 入口：`train.py`（ResNet-18，`--epochs` 默认 5，`--batch-size` 默认 128）
- 目标分区：`P107-RTX5090`（竞赛账户，由 `discover` 解析）
- 资源：1 GPU / 4 CPU / 16G / 2:00:00（从 `torch` import 推断）
- 环境：conda 环境 `resnet`（按项目目录名自动命名）

## 第 0 步：前置

已安装 lazy107（安装方式见 [README](../README.md#安装)：GitHub 克隆或 tar
发布包，随后 `pip install -e .`），并完成一次性平台探测：

```bash
lazy107 discover
```

```text
user=pb24061316
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
wrote /home/scc/pb24061316/.config/lazy107/config.toml
```

**说明了什么**：discover 从实时 Slurm 关联解析出账户有权使用的
account/partition/QoS 并写入用户级配置——之后所有项目自动继承，
不会再遇到 `Invalid account or account/partition combination`。

## 第 1 步：准备项目目录

把示例复制成自己的项目（目录名即 conda 环境名）：

```bash
mkdir -p ~/demos
cp -r examples/resnet-cifar10 ~/demos/resnet
cd ~/demos/resnet
ls
```

```text
data.py  model.py  train.py
```

**说明了什么**：项目只有三个 `.py` 文件。CIFAR-10 数据会在首次运行时由
torchvision 自动下载到 `data/`；若计算节点无法访问外网，先在登录节点
下载好，或用 GUI 上传 `data/` 目录（`lazy107 transfer` 会打印打包 →
上传 → 解包 → 校验的清单）。

## 第 2 步：看看工具对项目的理解

```bash
lazy107 check
```

```text
entry: train.py
env: not prepared — run `lazy107 env --yes`
runs: none recorded
commands: 1 recorded (latest: check)
next: lazy107 env --yes
```

**说明了什么**：`check` 一眼看清进度（入口已识别、环境未装、零提交记录）
和唯一正确的下一步。

## 第 3 步：查看解析后的运行计划

```bash
lazy107 plan
```

```text
entry=train.py
partition=P107-RTX5090
qos=qos_p107-rtx5090
account=competition
cpus=4
mem=16G
gpu=1
nodes=1
ntasks=1
time=2:00:00
log_dir=logs
conda_env=
job_name=
command=
array=
edit any value above in 107.toml — open the project folder in the Web Shell
GUI (Files) and edit 107.toml, then `lazy107 plan` to re-check
```

**说明了什么**：

- `gpu=1 cpus=4 mem=16G time=2:00:00` 不是谁写出来的——`plan` 扫描了
  `train.py` 里的 `import torch`（以及 data/model.py），推断出 GPU 档。
  项目没有任何依赖文件，这走的是纯 import 扫描路径。
- partition/qos/account 来自第 0 步 discover 写的全局配置。
- 最后一行指出：想改任何值，都在这一个 `107.toml` 文件里。

## 第 4 步：预览环境准备（只打印，不执行）

```bash
lazy107 env --dry-run
```

```text
# would write requirements.txt (imports: torch, torchvision, tqdm)
# would set conda_env = "resnet" in 107.toml
module load miniconda/py312  # no-op if conda is already on PATH
conda create -y -n resnet python=3.12
conda run -n resnet pip install -r requirements.txt
conda run -n resnet pip install torch torchvision  # PyPI default wheel bundles CUDA
```

**说明了什么**：

- `env` 检测到项目没有依赖文件，会**自动扫描代码 import 并生成
  requirements.txt**（`torch`、`torchvision`、`tqdm`）——第一行注释如实
  预告了这件事。
- 因为依赖里有 torch（GPU 档），命令序列**最后强制用 PyPI 默认源重装
  torch**：默认 wheel 自带 CUDA，conda 频道的 CPU 版永远不可能成为
  最终安装。
- `--dry-run` 只打印、什么都不写。

## 第 5 步：创建环境（一次性，约 10 分钟）

```bash
lazy107 env --yes
```

```text
wrote requirements.txt (imports: torch, torchvision, tqdm)
$ module load miniconda/py312  # no-op if conda is already on PATH
$ conda create -y -n resnet python=3.12
...（conda 创建输出）...
$ conda run -n resnet pip install -r requirements.txt
...（pip 安装输出）...
$ conda run -n resnet pip install torch torchvision  # PyPI default wheel bundles CUDA
...（pip 安装输出）...
torch CUDA build verified (torch.version.cuda=12.6)
wired conda_env = 'resnet' into 107.toml
```

**说明了什么**：

- `--yes` 才真正执行；每条命令先回显再运行，失败即停并报具体命令。
- 装完自动校验：`torch.version.cuda=12.6` 证明装的是 CUDA 版 torch
  （不用 `torch.cuda.is_available()`——它在登录节点恒为 False）。
- 环境自动接线：`conda_env = "resnet"` 写进了 `107.toml`，之后的作业
  脚本会自动激活它。

## 第 6 步：再看计划——环境已生效

```bash
lazy107 plan
```

```text
entry=train.py
partition=P107-RTX5090
qos=qos_p107-rtx5090
account=competition
cpus=4
mem=16G
gpu=1
nodes=1
ntasks=1
time=2:00:00
log_dir=logs
conda_env=resnet
...
```

**说明了什么**：`conda_env=resnet` 已在计划中。同一份 `plan` 输出就是
提交时将使用的全部事实。

## 第 7 步：预览将要提交的脚本

```bash
lazy107 render --dry-run
```

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate resnet
set -u

python - <<'PY'
import importlib.util
if importlib.util.find_spec('torch'):
    import torch
    assert torch.cuda.is_available(), 'GPU requested but CUDA unavailable in this job'
PY

python train.py
```

**说明了什么**：

- `--gres=gpu:1` 出现了——因为 `gpu=1`；`--account` 来自 discover；
  日志拆成 `.out`/`.err` 两份（traceback 落在 `.err`）。
- conda 激活块用官方 `conda.sh` 方式（`set +u` 保护）。
- 那段 heredoc 是 **CUDA 运行时守卫**：作业里如果 CUDA 不可用，会在
  任何训练代码运行前响亮失败——「申请了 GPU」永远不会静默变成 CPU 训练。
- `--dry-run` 只打印；`lazy107 render`（不带参数）才会写入
  `scripts/train.sbatch`。

## 第 8 步：提交

```bash
lazy107 submit --yes
```

```text
submitted job 53085 (train)
watch: lazy107 watch 53085
```

**说明了什么**：`submit` 内部依次完成了校验 → sinfo 预检 → 关联与上限
校验 → GPU 构建校验 → 渲染 → 确认 → 写脚本 → `sbatch` → 记账
（见 [design.md §13](design.md)）。每一步失败都会在提交前拦住你。

## 第 9 步：监控

```bash
lazy107 watch 53085
```

```text
scontrol show job 53085
squeue --me
tail -f logs/train_53085.out
tail -f logs/train_53085.err
```

```bash
tail -f logs/train_53085.out
```

```text
device=cuda seed=42 epochs=5 batch=128 data_dir=data
train=50000 test=10000
epoch   1 | train_loss 1.2345 train_acc 0.5632 | test_loss 0.9123 test_acc 0.6789 | lr 9.05e-02
epoch   2 | train_loss 0.6789 train_acc 0.7654 | test_loss 0.6543 test_acc 0.7765 | lr 6.55e-02
...
```

**说明了什么**：`device=cuda` 证明训练真的跑在 GPU 上；`watch` 给的
四条命令覆盖了状态查询和两个日志流。

## 第 10 步：作业完成——真实的训练结果

```bash
sacct -j 53085 --format=JobID,State,ExitCode,Elapsed
```

```text
JobID         State     ExitCode  Elapsed
53085         COMPLETED 0:0       00:04:12
```

```bash
cat outputs/metrics.json
```

```json
{
  "best_test_acc": 0.8156,
  "epochs": 5,
  "device": "cuda",
  "elapsed_s": 252.1
}
```

**说明了什么**：从「三个 .py 文件」到「COMPLETED + 真实
`best_test_acc` + GPU 训练时长」，全程只敲了 5 条命令，没有手写一行
Slurm 参数。检查点保存在 `outputs/best.pt`。

## 第 11 步：账本与进度

```bash
cat notes/runs.md
```

```markdown
## Job 53085

- **Submitted at**: 2026-09-05T02:13:11Z
- **Batch script**: scripts/train.sbatch
- **Entry point**: train.py
- **Account**: competition
- **Partition**: P107-RTX5090
- **QoS**: qos_p107-rtx5090
```

```bash
lazy107 check
```

```text
entry: train.py
env: 'resnet' ready
runs: 1 recorded (latest: 53085)
commands: 9 recorded (latest: submit --yes)
next: lazy107 watch 53085
```

**说明了什么**：每一次提交、每一条命令都有账可查——项目状态不需要
git 也能完整复现。

## 第 12 步：同一条命令走完全程（everything）

对下一个项目，上述 3–8 步可以合并成一条交互命令：

```bash
cd ~/demos/another-project
lazy107 everything
```

```text
existing conda environments:
  1) resnet
  2) base
reuse an env [1-2, name, Enter=install fresh]: 1
validated env 'resnet': all detected dependencies importable (torch CUDA build 12.6)
wired conda_env = 'resnet' into 107.toml

slurm resources (Enter keeps the current values):
  1) cpu-light  gpu=0 cpus=2 mem=4G time=1:00:00
  2) cpu-heavy  gpu=0 cpus=8 mem=32G time=8:00:00
  3) gpu-light  gpu=1 cpus=4 mem=16G time=2:00:00
  4) gpu-heavy  gpu=1 cpus=8 mem=64G time=12:00:00
  5) gpu-multi  gpu=2 cpus=16 mem=64G time=24:00:00
pick a preset [1-5], c to customize each field: 3

entry=train.py
...（完整计划）...

preview the sbatch script? [y/N] y
...（完整脚本）...

submit train on P107-RTX5090 (gpu=1)? [y/N] y
submitted job 53086 (train)
```

**说明了什么**：向导的每个决策点都有默认值（回车即接受）；环境复用
会**先校验再固定**（确认依赖齐全、CUDA 构建正确才写进 `107.toml`）；
资源用命名预设而非猜测；提交前的确认提示是最后一道闸。

## 第 13 步：失败了怎么办（预览 debug）

作业失败时，不需要自己翻日志：

```bash
lazy107 debug <job_id>
```

```text
job 53087: state=FAILED exit=1:0
cause: missing dependency 'nonexistent_module_xyz'
fix: add nonexistent_module_xyz to requirements.txt/environment.yml and re-run `lazy107 env --yes`
evidence: ModuleNotFoundError: No module named 'nonexistent_module_xyz'
```

**说明了什么**：`debug` 把 sacct 退出码和真实 107 日志签名两层证据翻译成
一行原因 + 一行修法。完整的失败场景演示见 [demos/04-debug.md](demos/04-debug.md)。

## 总结

这 13 步串起了 lazy107 的完整价值：

1. **零配置**（第 0 步）：discover 一次解决账户/分区/QoS；
2. **零知识**（第 3、7 步）：资源推断 + 确定性脚本，一行 Slurm 不用写；
3. **零妥协**（第 5、7 步）：GPU 即 CUDA，四道防线；
4. **零猜测**（第 12 步）：预设 + 复用 + 校验的问答式向导；
5. **零失忆**（第 11 步）：账本记录每一次提交；
6. **零甩锅**（第 13 步）：失败时工具告诉你原因和修法。

每一步的输出都可由 `scripts/capture-demos.sh` 在集群上重新采集
（见 [demos/INDEX.md](demos/INDEX.md)），保证文档与真实行为一致。
