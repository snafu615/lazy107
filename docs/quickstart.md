# 快速开始

5 分钟内提交第一个作业。所有操作都在**集群上**（登录节点或 Web Shell）
进行——本地无需安装任何东西。

## 前置条件

- 能登录 107 登录节点（SSH）或 Web Shell。
- 集群上有 Python ≥ 3.11（运行 lazy107 本身用）。
- 代码和数据已在集群上（见下文「代码和数据怎么上集群」）。

### 代码和数据怎么上集群

lazy107 不替你搬文件。任选其一或组合使用：

- **git（可选）**：在登录节点 `git clone` 你的仓库。工具从不自动碰 git；
  提交记录放在 `notes/runs.md` 里。
- **GUI 文件管理器**（SCOW / Xftp）：直接拖拽上传。传大文件前先运行
  `lazy107 transfer`，它会打印一份清单：

  ```text
  # 1. 本地打包（不放进 git）:
  # 2. 用 GUI 上传到登录节点 ~/transfer/
  # 3. 在集群上解包:
  # 4. 校验完整性（和本地 sha256 对比）:
  ```

  清单只打印、从不执行任何命令。

## 第 1 步：安装

拿到 lazy107 源码有两种方式，任选其一。安装目标：登录节点上任一
Python ≥ 3.11 环境（建议先 `conda activate` 常驻环境，或新建
`conda create -n lazy107 python=3.11`）。

**方式一：GitHub 克隆**

```bash
git clone https://github.com/snafu615/lazy107.git && cd lazy107
python -m pip install -e .            # 开发安装，改源码即生效
lazy107 --version
# -> lazy107 0.1.0
```

**方式二：tar 发布包**

把 `lazy107-0.1.0.tar.gz`（发布附件）上传到登录节点后解包：

```bash
tar -xzf lazy107-0.1.0.tar.gz && cd lazy107
python -m pip install -e .            # 开发安装
lazy107 --version
# -> lazy107 0.1.0
```

### 一次性平台配置

```bash
lazy107 discover --dry-run   # 先打印结果，不写任何文件
lazy107 discover             # 写入 ~/.config/lazy107/config.toml
```

```text
user=pb24061316
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
wrote /home/.../.config/lazy107/config.toml
```

这一步查询 `sacctmgr`/`scontrol`，解析出**你有权使用的** account/partition/QoS。
它解决的正是最常遇到的 `Invalid account or account/partition combination`：
内置默认值（`Students`/`qos_stu_default`）对竞赛账户并不可用，discover 会
换成对的组合。只做这一次，之后所有项目自动继承。

## 第 2 步：创建项目（或直接用已有项目）

```bash
lazy107 init my-project
cd my-project
```

`init` 从内置模板脚手架出：

```text
my-project/
├── src/
│   ├── data.py       # 数据加载（待实现）
│   ├── model.py      # 模型定义（待实现）
│   └── train.py      # 训练入口（待实现）
├── environment.yml   # python=3.12 + pip
├── pyproject.toml    # 依赖里带了 torch → 自动推断 GPU 档
├── scripts/train.sbatch
├── .gitignore
└── README.md
```

把 `src/data.py`、`src/model.py`、`src/train.py` 填上你的代码即可。
已有项目则什么都不用做——入口文件自动检测：`train.py` > `main.py` > 其他 `.py`。

### 想一步到位？`lazy107 everything`

```bash
lazy107 everything
```

以「问 → 答 → 继续」的方式走完全部流程，每一步都有默认值（回车即接受）：

1. 入口文件有多个时，编号选择（回车 = 推荐项），选择会固定进 `107.toml`；
2. 环境：列出已有 conda 环境供复用（逐个校验依赖与 CUDA 构建），或新装一个；
3. Slurm 资源：5 档预设（`cpu-light`/`cpu-heavy`/`gpu-light`/`gpu-heavy`/
   `gpu-multi`）任选其一，或按 `c` 逐字段修改 `gpu`/`cpus`/`mem`/`time`；
4. 打印解析后的 plan（此时会做完整校验）；
5. 可选：预览 sbatch 脚本；
6. 最终确认后提交。

`--yes` 跳过所有交互（新装环境、保留当前资源、直接提交）；
`--entry` 预先指定入口文件。

## 第 3 步：准备 conda 环境

```bash
lazy107 env --dry-run   # 先打印——这一步永远安全
lazy107 env --yes       # 确认无误后执行
```

对脚手架项目，`--dry-run` 大致打印：

```bash
module load miniconda/py312  # conda 已在 PATH 上时为空操作
conda create -y -n my-project python=3.12
conda env update -n my-project -f environment.yml
conda run -n my-project pip install -e .
conda run -n my-project pip install torch torchvision  # CUDA wheel
```

要点：

- 项目依赖 torch → 推断为 GPU 档 → 最后一步强制用 PyPI 默认源装
  `torch`/`torchvision`（默认 wheel 自带 CUDA）；jax 项目会装
  `jax[cuda12]`。conda 频道里的 CPU 版 torch 永远不是 GPU 作业的最终安装。
- **只有代码、没有任何依赖文件的项目同样能处理**：`env` 扫描 `.py` 的
  import（标准库和项目自身模块自动跳过），把第三方包写进
  `requirements.txt`（之后可以手动改版本），再安装。
- 装完后自动校验：GPU 作业会打印
  `torch CUDA build verified (torch.version.cuda=12.6)`。
- `env --yes` 会把环境写进 `107.toml` 的 `conda_env`（没有则创建该文件），
  之后渲染的作业脚本会自动 `conda activate` 它。

### 已有环境？复用，不重装

在终端里直接运行（不带任何参数）：

```bash
lazy107 env
```

它会列出你已有的 conda 环境，选中一个后自动校验：项目 import 的每个包在
该环境里是否都能找到（`importlib.util.find_spec` 探测），GPU 作业还会校验
torch 是不是 CUDA 版。校验通过就把它固定为 `conda_env`，不装任何新包；
不通过会告诉你缺了哪些包，重选或回车进入全新安装流程。

## 第 4 步：查看计划与脚本

```bash
lazy107 plan
```

对脚手架项目（且已 discover）：

```text
entry=src/train.py
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
conda_env=my-project
job_name=
command=
array=
edit any value above in 107.toml — ...
```

`gpu=1`、`cpus=4`、`mem=16G`、`time=2:00:00` 是从 `torch` 依赖推断出来的；
partition/qos/account 来自 discover 写的全局配置。想改任何值：打开 Web Shell
的「文件」面板编辑 `107.toml`，再跑一次 `lazy107 plan` 立即生效。

> 项目里有多个 `.py` 时，`plan` 会列出编号菜单让你选入口（回车 = 推荐项），
> 选择结果固定进 `107.toml` 的 `entry`，之后所有命令都遵循它。

提交前看一眼脚本（只打印、不写文件）：

```bash
lazy107 render --dry-run
```

检查两件事：`#SBATCH --gres=gpu:1` 在不在（说明 torch 被检测到了），
以及脚本里的 CUDA 运行时断言在不在。核心内容：

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --account=competition
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

# conda 激活块（conda_env 已设置时）
# CUDA 运行时断言（gpu > 0 时）
python src/train.py
```

## 第 5 步：提交

```bash
lazy107 submit --yes
```

`submit` 依次执行：

1. 校验计划（入口存在、各字段安全）；
2. `sinfo` 预检分区是否存在（`--skip-check` 可跳过）；
3. account/partition/QoS 关联校验 + 时长/内存上限校验（发现超限直接拦截）；
4. `gpu>0` 时校验环境的 torch 是 CUDA 版，CPU-only 则**拦截**；
5. 渲染并写入 `scripts/train.sbatch`（旧文件自动备份为 `.old`）；
6. `sbatch` 提交并捕获作业号；
7. 追加记录到 `notes/runs.md`。

成功输出：

```text
submitted job 53085 (train)
watch: lazy107 watch 53085
```

## 第 6 步：监控

```bash
lazy107 watch <job_id>     # 打印下面四条命令
scontrol show job <job_id>
squeue --me
tail -f logs/train_<job_id>.out
tail -f logs/train_<job_id>.err   # 报错信息（traceback）落在这里
```

也可以直接看队列（`lazy107 status`）或只看日志命令（`lazy107 logs <job_id>`）。

## 第 7 步：失败了？让它告诉你原因

```bash
lazy107 debug <job_id>
```

结合两层证据给出诊断：sacct 退出码语义（`0:9` = 被 SIGKILL，通常是内存不足；
`0:15` = 被 SIGTERM，通常是超时或手动取消；`1:0` = 程序自身报错）+
从真实 107 日志采集的失败签名（缺模块、缺数据、超时行、CUDA OOM……）。

```text
job 53073: state=FAILED exit=1:0
cause: missing dependency 'nonexistent_module_xyz'
fix: add nonexistent_module_xyz to requirements.txt/environment.yml and re-run `lazy107 env --yes`
evidence: ModuleNotFoundError: No module named 'nonexistent_module_xyz'
```

## 随时查看进度

```bash
lazy107 check
```

```text
entry: src/train.py
env: 'my-project' ready
runs: 1 recorded (latest: 53085)
commands: 12 recorded (latest: submit --yes)
next: lazy107 watch 53085
```

## 常见问题

### `env` 报 conda 找不到

```text
lazy107: conda not found: run `module load miniconda/py312` or install under ~/miniconda3
```

在 shell 里执行 `module load miniconda/py312`，或在 `~/miniconda3` 安装 Miniconda。

### 提交被拦截：torch 是 CPU 版

```text
GPU requested but the env's torch is a CPU-only build; reinstall via `lazy107 env --yes`
```

环境里的 torch 来自 CPU 源（如 conda-forge）。用 `lazy107 env --yes` 重装——
PyPI 默认 wheel 自带 CUDA。

### 提交报 Invalid account or account/partition combination

你的账户无权使用当前配置的分区/QoS（例如竞赛账户配了默认的
`Students`/`qos_stu_default`）。解决：跑一次 `lazy107 discover`，
或在 `107.toml` 里手动设置 `account`/`partition`/`qos`。

### `plan` / `render` 报 entry 相关错误

没找到 `train.py`/`main.py`。显式指定入口：

```bash
lazy107 submit --entry src/other.py --yes
```

### 分区预检失败

```text
partition <name> failed preflight; use --skip-check to override
```

分区名写错了或当前不可用。用 `sinfo -p <name> -h` 确认，改 `107.toml`，
或确实需要时用 `--skip-check` 跳过。

## 验收清单

- [ ] `lazy107 --version` 在登录节点可用
- [ ] `lazy107 init my-project` 创建了预期的文件
- [ ] `lazy107 check` 打印入口/环境/记录状态与单条 `next:` 提示
- [ ] `lazy107 plan` 对 torch 项目显示 `gpu=1`
- [ ] `lazy107 render --dry-run` 打印的脚本带 `--gres=gpu:1` 和 CUDA 断言
- [ ] `lazy107 submit --yes` 输出 `submitted job <job_id>`
- [ ] `squeue -u $USER` 能看到作业
- [ ] `notes/runs.md` 里有该作业号
- [ ] `lazy107 debug <job_id>` 能对失败作业给出 cause + fix
- [ ] `lazy107 transfer` 打印有效清单（或 "No large files found"）
- [ ] `lazy107 everything` 能走完 环境 → 资源预设 → 提交 的问答流程

## 附：快速卸载

在装有 lazy107 的环境里执行（先 `conda activate` 该环境）：

```bash
python -m pip uninstall -y lazy107    # 移除包与 lazy107 命令
rm -rf ~/.config/lazy107              # 用户级配置（discover 写入）
conda env remove -n lazy107 -y        # 若建过同名专用环境
rm -rf ~/.lazy107                     # 若用 venv 方式（--prefix）装过
```

最后按需删除克隆/解包出来的 `lazy107/` 源码目录。项目目录里的
`107.toml`、`notes/`、`logs/`、`outputs/` 是项目数据而非工具本身，
是否删除取决于项目需要。
