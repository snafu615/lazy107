# lazy107

lazy107 是面向中国科大 107 计算平台的**集群端命令行工具**：自动识别入口文件、
从依赖推断资源（GPU/CPU/内存/时长）、准备 conda 环境、生成 sbatch 脚本、提交作业并记账。
不需要背任何 Slurm 参数，本地也无需安装任何东西——全部在登录节点 / Web Shell 上完成。

## 它能解决什么问题

| 你遇到的问题 | lazy107 的做法 |
|---|---|
| `Invalid account or account/partition combination` | `lazy107 discover` 从实时 Slurm 关联解析出你有权使用的 account/partition/QoS，写入用户级配置，一次解决 |
| 不知道申请多少资源 | 从依赖自动推断（torch/jax/tf → 1 GPU/4 CPU/16G/2h；否则 CPU 档），`everything` 还提供 5 档预设 |
| 每个项目都要重新配一遍环境 | `lazy107 env` 优先**复用已有 conda 环境**（自动校验依赖是否齐全），缺了才新装 |
| 作业挂了看不懂日志 | `lazy107 debug <job_id>` 一行给出原因（cause）+ 修法（fix）+ 证据（evidence） |
| 申请了 GPU 却跑在 CPU 上 | 四道防线保证 GPU 请求永远等于 CUDA（见下） |
| 多卡 / 调参作业不会写 | 检测到 DDP 代码自动改用 torchrun 启动；`array = "1-4%2"` 一键作业数组 |
| 忘了自己提交过什么 | 每次提交自动记入 `notes/runs.md`，每条命令记入 `notes/history.md` |

## 快速开始（5 条命令）

```bash
# 1. 一次性平台探测：解析你有权使用的 account/partition/QoS
lazy107 discover --dry-run
lazy107 discover

# 2. 脚手架一个新项目（已有项目则直接 cd 进去）
lazy107 init my-project
cd my-project

# 3. 准备 conda 环境（默认只打印；--yes 才执行）
lazy107 env --dry-run
lazy107 env --yes

# 4. 查看将提交什么，然后提交
lazy107 plan
lazy107 render --dry-run
lazy107 submit --yes

# 5. 监控 / 诊断
lazy107 watch <job_id>
lazy107 debug <job_id>    # 作业失败时：sacct 退出码 + 日志签名 → 原因/修法
```

不想分步？`lazy107 everything` 用问答方式把「选入口 → 配环境 → 选资源 → 预览 → 提交」
一条命令走完，每个决策点都有提示，按回车即用当前值。

完整分步教程见 [docs/quickstart.md](docs/quickstart.md)，
端到端实例见 [docs/demo.md](docs/demo.md)，
内部设计见 [docs/design.md](docs/design.md)。

## 主要功能

- **零配置上手**：`discover` 查询 `sacctmgr`/`scontrol`，把你有权使用的
  account/partition/QoS 写入 `~/.config/lazy107/config.toml`，提交不再报
  "Invalid account or account/partition combination"。
- **从代码推断资源**：依赖里出现 torch/jax/tensorflow → 自动升级为
  GPU 档（1 GPU/4 CPU/16G/2h）；否则 CPU 档（2 CPU/4G/1h）。
  没有依赖文件也没关系——`env` 会扫描 `.py` 的 import，写出
  `requirements.txt` 再安装（`sklearn`→`scikit-learn` 这类名不副实的包已内置映射）。
- **环境复用优先**：终端里直接运行 `lazy107 env`，它会列出你已有的 conda
  环境，逐个校验「项目 import 的包是否都能找到、GPU 作业的 torch 是否是
  CUDA 版」，选一个就直接复用，不重装。
- **GPU 即 CUDA，四道防线**：安装时锁定 CUDA 版 wheel → 装完校验
  `torch.version.cuda` → 提交前拦截 CPU-only 构建 → 生成的脚本里带运行时
  CUDA 断言。任何一条被破坏，作业都会响亮地失败，而不是悄悄在 CPU 上跑。
- **确定性脚本生成**：`plan` 是唯一事实来源；同一份配置永远渲染出同一份
  sbatch。检测到 DDP 代码自动切换 `torchrun --standalone
  --nproc_per_node=$SLURM_GPUS_ON_NODE` 启动；多节点 DDP、作业数组
  （`--array`，每个任务独立日志）自动生成。
- **失败自动诊断**：`debug` 基于两层证据——sacct 退出码语义
  （`0:9` OOM、`0:15` 超时）和从真实 107 日志里采来的失败签名
  （缺模块、缺数据、超时行、CUDA OOM……），按「最具体优先」匹配，输出
  cause/fix/evidence 三行。
- **安全边界**：`--dry-run` 只打印、绝不写文件；提交前必须确认（`--yes`
  跳过）；`transfer` 只打印 GUI 上传清单，从不替你传文件或删文件；
  登录节点只做管理，训练永远走 `sbatch`。
- **全程记账**：每次提交追加到 `notes/runs.md`（作业号 + 生效参数），
  项目内每条命令追加到 `notes/history.md`；`lazy107 check` 随时告诉你
  当前进度和下一步该做什么。

## 安装

lazy107 是零运行时依赖的纯 Python 包（Python ≥ 3.11），只装在**登录节点**上。
下面两条获取源码的方式任选其一；安装目标为登录节点上任一 Python ≥ 3.11 环境
（建议先 `conda activate` 常驻环境，或新建 `conda create -n lazy107 python=3.11`）。

**方式一：GitHub 克隆（推荐，跟随最新版本）**

```bash
git clone https://github.com/snafu615/lazy107.git && cd lazy107
python -m pip install -e .            # 开发安装，改源码即生效
lazy107 --version
```

**方式二：tar 发布包（离线友好）**

把 `lazy107-0.1.0.tar.gz`（发布附件）上传到登录节点后解包：

```bash
tar -xzf lazy107-0.1.0.tar.gz && cd lazy107
python -m pip install -e .            # 开发安装
# 完全离线时（无法联网拉取构建工具），改用压缩包内自带的 wheel：
python -m pip install dist/lazy107-0.1.0-py3-none-any.whl
```

验证：

```bash
lazy107 --version
lazy107 --help    # 列出全部 14 个子命令
```

## 卸载（快速）

在装有 lazy107 的环境里执行（先 `conda activate` 该环境）：

```bash
python -m pip uninstall -y lazy107    # 移除包与 lazy107 命令
rm -rf ~/.config/lazy107              # 用户级配置（discover 写入）
conda env remove -n lazy107 -y        # 若建过同名专用环境
rm -rf ~/.lazy107                     # 若用 venv 方式（--prefix）装过
```

克隆/解包出来的 `lazy107/` 源码目录按需删除；项目里的 `107.toml`、
`notes/`、`logs/`、`outputs/` 是项目数据而非工具本身，按需保留。

## 命令一览

| 命令 | 作用 |
|---|---|
| `lazy107 init <name>` | 用内置模板脚手架一个新项目 |
| `lazy107 plan [--entry] [--array]` | 打印解析后的运行计划；入口文件有多个时交互选择并固定到 `107.toml` |
| `lazy107 render [--entry] [--array] [--dry-run]` | 写出（或打印）`scripts/<入口>.sbatch` |
| `lazy107 env [--name] [--entry] [--dry-run] [--yes]` | 打印/执行 conda 环境准备；交互模式下优先复用已有环境 |
| `lazy107 submit [--entry] [--array] [--dry-run] [--skip-check] [--yes]` | 校验 → 预检 → 渲染 → 确认 → 提交 → 记账 |
| `lazy107 watch <job_id> [--job-name]` | 打印监控命令（scontrol/squeue/tail） |
| `lazy107 status [job_id]` | 打印队列状态命令 |
| `lazy107 logs <job_id> [--job-name]` | 打印查看 stdout + stderr 日志的命令 |
| `lazy107 debug <job_id> [--job-name]` | 诊断失败作业：sacct 退出码 + 日志签名 → 原因/修法 |
| `lazy107 transfer` | 打印大文件 GUI 上传清单（只打印，不执行） |
| `lazy107 discover [--dry-run]` | 解析你有权使用的 account/partition/QoS 并写入用户级配置 |
| `lazy107 config [--init]` | 查看生效配置，或写出 `107.toml` |
| `lazy107 check` | 显示工作流状态（入口/环境/记录）与下一步 |
| `lazy107 everything [--entry] [--yes]` | 问答式一键全流程：入口 → 环境 → 资源预设 → 预览 → 提交 |

## 配置

五层合并，**后者覆盖前者**：

```text
工具默认值 < 依赖推断 < ~/.config/lazy107/config.toml（discover 写入）
           < 项目 107.toml < LAZY107_* 环境变量 < 命令行参数
```

项目级 `107.toml`（任何字段都可以省略；`lazy107 config --init` 生成完整模板）：

```toml
entry = ""              # 入口文件（留空 = 自动检测 train.py > main.py > 其他 .py）
partition = "Students"  # discover 后由全局配置覆盖为你的分区
qos = "qos_stu_default"
account = ""            # 留空 = Slurm 默认账户；非空才渲染 --account 行
cpus = 4
mem = "16G"
gpu = 1
nodes = 1               # >1 → 多节点分配
ntasks = 1              # 多节点 DDP 时必须等于 nodes
time = "2:00:00"
log_dir = "logs"
conda_env = "my-project"  # 作业脚本会 conda activate 它
job_name = ""             # 留空 = 入口文件名
command = ""              # 设置后按原样执行（取代 python/torchrun）
array = ""                # 设置后变成作业数组，如 "1-5%2" 或 "0,2,4"
```

改完 `107.toml` 再跑一次 `lazy107 plan` 即可看到效果——每个命令的输出末尾都会
提示你「要改的值都在这一个文件里」。

## 文档导航

| 文档 | 读者 | 内容 |
|---|---|---|
| [intro.md](intro.md) | 评委、任何人 | 作品简介：背景、解决的问题、核心功能、模型与 API 说明、创新点 |
| [docs/quickstart.md](docs/quickstart.md) | 新用户 | 从安装到第一个作业的分步教程 |
| [docs/demo.md](docs/demo.md) | 所有人 | 端到端完整实例：ResNet-18/CIFAR-10 真实训练 |
| [docs/design.md](docs/design.md) | 开发者、评委 | 架构、代码结构、各模块功能、安全边界 |
| [docs/demos/INDEX.md](docs/demos/INDEX.md) | 评委、新用户 | 六个实机演示（英文，待翻译） |
| [docs/runbook.md](docs/runbook.md) | 验收测试 | 分步验收清单与常见阻塞（英文，待翻译） |
| [docs/env-pipeline.md](docs/env-pipeline.md) | 开发者 | 环境流水线详解：分析 → 生成 → 执行 → 消费（英文，待翻译） |

## 许可

MIT License。
