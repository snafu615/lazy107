# lazy107 设计文档

本文档介绍 lazy107 的定位、设计思路、技术架构、代码结构、各模块职责、核心机制
（配置合并、脚本渲染、失败诊断、GPU/CUDA 完整性）与安全边界。
读者：开发者、维护者、评委。评审速览：先看 §1 的决策表与「技术难点与应对」小节，
再按需深入对应章节。

## 1. 定位

lazy107 是**纯集群端**的命令行工具：只运行在 107 登录节点 / Web Shell 上，
负责把「一个项目目录」变成「一个成功提交并记录在案的 Slurm 作业」。
本地不需要装任何东西；训练永远通过 `sbatch` 在计算节点上执行，
登录节点只做管理。

已锁定的设计决策：

| 决策 | 含义 |
|---|---|
| 集群端运行 | 零本地安装；代码/数据经 git（可选）或 GUI 文件管理器上传 |
| 全 conda、无 uv | 环境 = `conda create -n <项目> python=3.12` + pip 安装；批处理脚本用官方 `source "$(conda info --base)/etc/profile.d/conda.sh"` 方式激活 |
| 不替用户搬文件 | `transfer` 只打印 GUI 上传清单（打包 → 上传 → 解包 → 校验），从不执行任何传输命令 |
| git 可选 | 工具从不调用 git；提交记录放在 `notes/runs.md` |
| 配置文件优先 | `107.toml` + 用户级 `~/.config/lazy107/config.toml` + `LAZY107_*` 环境变量 + 命令行参数 |
| 零运行时依赖 | 仅标准库（`tomllib`、`dataclasses`、`subprocess`、`importlib.resources` 等）；Python ≥ 3.11 |
| 资源显式选择 | 不做负载估算（batch/epoch/模型大小一概不猜）；推断给出保守默认，`everything` 提供命名预设，超了就用 `debug` 的退出码提示按证据加码 |

### 设计思路（为什么这么做）

- **把高频易错动作收敛为可复现流程**：一次作业提交涉及 Slurm 参数、环境、
  入口、资源、日志五个知识域，任何一个出错都只暴露在作业失败之后。lazy107
  把这一过程收敛成「项目目录 → RunPlan → 渲染 → 提交」的确定性流水线，
  任何一步都可在提交前用 `--dry-run` 完整预览。
- **确定性优于猜测**：资源推断只做保守的二元启发式（有 GPU 框架 / 没有），
  绝不假装知道 batch size、epoch、模型大小；「猜错」的代价由 `debug` 的
  退出码语义 + 日志签名闭环弥补（0:9 → 加 mem，0:15 → 加 time）。
- **证据驱动，不信惯例**：平台行为（分区/QoS 权限、日志前缀格式、`--mem`
  是否强制执行）一律以实机采集为准，写进文档与测试（§18），不做假设。
- **失败要响亮、要可定位**：静默退化（申请了 GPU 却在 CPU 上跑、依赖缺失、
  环境未激活）是头号敌人，因此有 §14 四道防线和 §8 两层诊断；
  出现 traceback 意味着是 bug 而不是用户错误（§17）。
- **默认安全**：注入面全部净化、`--dry-run` 只读、`transfer` 只打印、
  交互确认是最后一道闸（§17）；配置文件写回以最小化方式追加（§4）。
- **纯函数分层 + 零运行时依赖**：`core/*` 无 I/O、`cluster/*` 独占
  subprocess，274 个用例全部离线可跑（§2、§19），工具本体仅用标准库
  （Python ≥ 3.11），安装与卸载各两条命令。

### 技术难点与应对（评委速览）

| 难点 | 难在哪 | 应对方案（章节） |
|---|---|---|
| 平台输出与文档/惯例不符 | `sacctmgr` 文本列会被截断；`scontrol` 把多个字段打包在同一行；新版 Slurm 日志去掉了 `slurmstepd:` 前缀并带 ISO 时间戳；`--mem` 实测未强制 | 一律用机器格式 `--parsable2 --noheader` + 整行正则解析；匹配器前缀无关；平台事实全部实机采集并记录在案（§12、§18） |
| 每个进 sbatch 的用户字段都是注入面 | 命令注入、伪造 `#SBATCH` 行可让工具提交任意负载 | 白名单校验拒绝空白与 shell 元字符，`validate_plan` 统一把关，新命令自动继承（§9） |
| 「申请了 GPU 却跑在 CPU 上」难以察觉 | conda 默认源装出的 torch 是 CPU 版；作业静默跑完，结果看起来正常 | 四道防线：安装时锁定 pip 默认源 → 装后查 `torch.version.cuda` → 提交前拦截 → 脚本内置运行时 CUDA 守卫（§14） |
| 失败原因只能事后从日志里人肉找 | 退出码 0:9/0:15 含义、日志签名随平台演进而变 | 两层诊断：sacct 退出码语义表 + 从真实 107 日志采集的签名库，最具体优先匹配，输出 cause/fix/evidence（§8） |
| 作业名与日志路径漂移、数组任务日志混写 | 多任务并发写同一文件，日志互相覆盖 | job-name 与日志文件名同源；数组作业用 `%x_%A_%a` 按主作业号+下标分文件（§3、§7） |
| conda 在 sbatch 批处理环境里的激活方式脆弱 | 交互 shell 的 `.bashrc` 初始化在批处理作业中不生效 | 用官方 `source "$(conda info --base)/etc/profile.d/conda.sh"` 激活；`env` 命令序列可打印、幂等、带超时（§12、§18） |
| 工具要自动改写用户配置文件 | 覆盖用户手写内容即不可接受 | `wire_*` 正则替换/追加的最小化写回，非空值默认保留；交互输入先净化（§4） |
| 无集群环境下如何保证行为正确 | 测试需要真实 Slurm | 分层架构：`core/*` 纯函数直接单测，`cluster/*` 以注入/fake 覆盖，274 个用例全部离线（§2、§19） |

## 2. 总体架构

三层结构，依赖方向严格单向：

```text
┌─────────────────────────────────────────────────────┐
│  编排层  cli.py（14 个子命令 + 交互辅助）            │
│         wizard.py（everything 向导，编排 cmd_*）     │
├─────────────────────────────────────────────────────┤
│  纯逻辑层 core/（无 subprocess、无集群概念）          │
│  plan defaults detect diagnose workflow validate     │
│  render record notebook                              │
├─────────────────────────────────────────────────────┤
│  I/O 层  cluster/（唯一执行外部命令的地方）           │
│  env discover monitor submit                         │
│  + manifest.py（配置合并与 107.toml 写回）           │
│  + template.py / transfer.py（模板与上传清单）       │
└─────────────────────────────────────────────────────┘
```

分层规则：

- `core/*` 是纯函数/纯数据：不 import `subprocess` 依赖的东西、不做文件系统之外
  的 I/O。集群探测（环境是否存在）以 callable 注入，保证单测无需集群。
- `cluster/*` 拥有全部 `subprocess` 调用，并统一做「命令不存在/超时/解析失败」
  的降级。
- `cli.py` 是唯一有用户 I/O 的地方（打印、交互提示）。
- `render_sbatch(plan, ddp)` 是 RunPlan 的**确定性纯函数**：同一份计划永远
  渲染出同一份脚本——这正是 `--dry-run` 值得信任的原因。

### 目录结构

```text
src/lazy107/
├── cli.py            # argparse 表面：14 个命令、_fail/_entry/交互辅助
├── wizard.py         # `everything` 向导（懒加载，编排 cli.cmd_*，不重复实现逻辑）
├── manifest.py       # 配置合并 + global_config_path + wire_entry/wire_fields/wire_conda_env
├── template.py       # 内置模板解析（importlib.resources，wheel 安全）
├── transfer.py       # 大文件发现 + 只打印的上传清单
├── core/
│   ├── plan.py       # RunPlan：一次提交的全部决策，单一事实来源
│   ├── defaults.py   # 工具默认值 + 依赖推断 + 代码 import 扫描 + DDP 检测
│   ├── detect.py     # 入口检测（train.py > main.py > 其他 .py > .ipynb）
│   ├── diagnose.py   # 失败诊断（纯逻辑：退出码 + 日志签名）
│   ├── validate.py   # 计划校验（注入安全、array 语法、多节点 DDP 约束）
│   ├── workflow.py   # 工作流状态检查（错序/漏步）+ check 仪表盘
│   ├── render.py     # sbatch 渲染（RunPlan → 脚本字符串）
│   ├── notebook.py   # .ipynb → .py（nbconvert 可选，JSON 兜底）
│   └── record.py     # notes/runs.md 账本 + notes/history.md 命令史
├── cluster/
│   ├── env.py        # conda 发现、环境存在性、命令序列、GPU 锁定、CUDA 构建校验
│   ├── discover.py   # sacctmgr/scontrol 解析、account/partition/QoS 解析、提交时校验
│   ├── monitor.py    # watch/logs 命令生成、sacct 查询、日志收集
│   └── submit.py     # sbatch/sinfo 执行与 SubmitError
└── templates/default/   # 包内数据（wheel 安全）：9 个脚手架文件
```

## 3. 数据模型：RunPlan

`core/plan.py` 的 `RunPlan`（frozen dataclass）是一次提交的**唯一事实来源**，
16 个字段：

```text
entry partition qos account cpus mem gpu nodes ntasks
time log_dir conda_env job_name command array
```

要点：

- `effective_job_name = job_name 或入口文件 stem`——sbatch 的 `--job-name` 与
  日志文件名永远使用同一个值（修复过「作业名与日志路径漂移」的缺陷）。
- `account` 非空才渲染 `#SBATCH --account=...` 行。
- `nodes`/`ntasks`（默认 1）决定分配规模；多节点 DDP 时 `ntasks` 必须等于
  `nodes`（校验器强制）。
- `command` 非空时**原样替换**启动行（最高优先级，完全用户掌控）。
- `array` 非空时（如 `1-5%2`、`0,2,4`）作业变成作业数组。

## 4. 配置合并（5 层）

`manifest.py::resolve_plan` 的合并顺序，**后者覆盖前者**：

```text
RunPlan 默认值 < 依赖推断(derive_defaults) < ~/.config/lazy107/config.toml
             < 项目 107.toml < LAZY107_* 环境变量 < CLI 参数
```

细节：

- `107.toml` 支持扁平键或 `[run]` 段两种写法（`load_manifest` 自动识别）。
- 用户级全局配置由 `lazy107 discover` 写入，只放平台身份值
  （account/partition/qos）；资源参数始终是项目级。
- 环境变量镜像字段名：`LAZY107_GPU=2`、`LAZY107_TIME=1:00:00` 等；
  数值字段（cpus/gpu/nodes/ntasks）做 int 强制转换。
- 全局配置文件缺失或损坏时静默忽略，不影响任何命令。

### 107.toml 写回机制（wire_*）

三个写回函数保证「工具自动修改配置文件」时绝不破坏用户手写内容：

- `wire_entry` / `wire_conda_env` / `wire_fields` 都是「正则替换已有键，否则
  追加一行」；文件不存在时创建一个最小文件。
- `wire_conda_env(force=False)` 保留用户已设置的非空值（只报告差异）；
  `force=True` 覆盖——用于「用户选择复用某个环境」的场景。
- 所有写回值都来自已净化的输入（交互选择器拒绝引号/反斜杠；环境名经
  `sanitize_env_name` 过滤为 `[A-Za-z0-9_.-]`），TOML 字符串无需转义。

## 5. 依赖推断与资源默认值（core/defaults.py）

### 依赖读取

`detect_dependencies()` 合并四个来源：

1. `pyproject.toml`（`[project].dependencies` + 所有 optional-dependencies 组）
2. `requirements.txt`
3. `environment.yml`（`- 包名` 行）
4. **代码 import 扫描**（`scan_imports`）

`scan_imports` 的规则：遍历项目内所有 `.py` 与 `.ipynb` 代码单元格；
标准库和项目自身模块跳过；隐藏目录（`.venv`/`.git` 等）不扫；
无法解码的文件（如 GBK）跳过而非报错；`import a, b` 的逗号形式完整捕获。
import 名到 pip 名只有少数不对应，由 `KNOWN_PACKAGES` 修正
（`sklearn`→`scikit-learn`、`skimage`→`scikit-image`、`PIL`→`Pillow`、
`cv2`→`opencv-python`、`yaml`→`PyYAML`），其余按自身名字安装。

### 资源档位

`derive_defaults` 是**二元启发式**，不是负载估算：

| 条件 | 结果 |
|---|---|
| 依赖含 GPU 框架（torch/pytorch/torchvision/torchaudio/tensorflow/tf-keras/keras/jax/jaxlib/flax） | `gpu=1 cpus=4 mem=16G time=2:00:00` |
| 否则 | `gpu=0 cpus=2 mem=4G time=1:00:00` |

设计立场：工具不猜 batch size、epochs、模型大小——那些只有用户知道。推断
给出保守起点，`everything` 的预设给出升级路径，`debug` 的退出码诊断给出
「按证据加码」的闭环。推断结果永远低于显式配置（`107.toml`、环境变量、
参数），绝不覆盖用户写下的值。

### DDP 检测

`detect_ddp()` 扫描同样一批代码文件，只认高置信度形式
（`DDP_PATTERNS` 共 8 条）：

```text
import/from torch.distributed
from torch.nn.parallel import DistributedDataParallel
DistributedDataParallel(...)
init_process_group(...)
import/from accelerate / deepspeed
strategy = "ddp" / 'ddp'
ddp_find_unused_parameters =
```

每行先剥掉 `#` 注释再匹配——**注释和 docstring 永远不会触发 torchrun**。
理由：误报的代价是把入口在 torchrun 下多进程启动 N 次；漏报的代价只是
退化为单进程，两者风险不对称，所以取高置信度。

## 6. 入口检测（core/detect.py）

优先级：`train.py` > `main.py` > 其余 `.py`（按路径排序）> `.ipynb`。
`recommend_entry` 取第一个。多候选且用户没指定时，`plan` 在 TTY 下弹出
编号菜单（回车 = 推荐项，可输入序号或文件名），选择通过 `wire_entry`
固定进 `107.toml`，之后所有命令一致遵循。选择顺序：
`--entry` 参数 > `107.toml` 的 `entry` > 自动检测。

## 7. sbatch 渲染（core/render.py）

`render_sbatch(plan, ddp)` 输出的脚本结构：

```bash
#SBATCH --job-name=<effective_job_name>
#SBATCH --account=<account>            # 仅 account 非空
#SBATCH --partition=...  --qos=...  --nodes=...  --ntasks=...
#SBATCH --ntasks-per-node=1            # 仅多节点 DDP
#SBATCH --array=<spec>                 # 仅 array 非空（已按 Slurm 语法校验）
#SBATCH --cpus-per-task=...  --mem=...  --gres=gpu:N  --time=...
#SBATCH --output=logs/%x_%j.out        # 数组作业换成 %x_%A_%a
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# [conda 激活块]  [CUDA 运行时守卫]  [数组任务提示]
<启动行>
```

启动行三选一（先匹配先赢）：

1. `command` 非空 → **原样使用**（完全用户掌控）；
2. DDP 且 `gpu>0`：
   - 单节点：`torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE <entry>`
   - 多节点：`srun torchrun --nnodes=$SLURM_JOB_NUM_NODES
     --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$SLURM_JOB_ID
     --rdzv_backend=c10d --rdzv_endpoint=... <entry>`（PyTorch 官方 Slurm 配方）
3. 否则 `python <entry>`。

其他细节：

- conda 激活块包在 `set +u` / `set -u` 之间（conda 脚本会触碰未定义变量）。
- 数组作业的日志用 `%x_%A_%a`（主作业号 + 任务下标），并发任务日志永不
  混写；脚本里附注释提示入口读取 `$SLURM_ARRAY_TASK_ID` 选择超参数行。
- `write_sbatch` 写入前把旧脚本备份为 `train.sbatch.old`、`.old2`……
- 每个写进脚本的自由字段（entry/partition/qos/account/mem/time/log_dir/
  conda_env/job_name）都先通过注入安全校验（见 §9）。

## 8. 失败诊断（core/diagnose.py + cluster/monitor.py）

`lazy107 debug <job_id>` 按「日志签名优先、退出码兜底」两层模型工作。

### 第一层：sacct 退出码语义

| ExitCode | 含义 | 修法提示 |
|---|---|---|
| `1:0` | 程序自身非零退出 | 读 `.err` 里的 traceback |
| `0:9` | 被 SIGKILL（通常内存不足） | 提高 `mem` 或减少内存占用 |
| `0:15` | 被 SIGTERM（超时或手动取消） | 提高 `time`、加 checkpoint 或减负载 |

### 第二层：日志签名（正则，从真实 107 日志采集）

按「最具体优先」排列，命中即胜：

1. `ModuleNotFoundError: No module named '<名>'` → 缺依赖，给出包名；
2. `FileNotFoundError: [Errno 2] ... '<路径>'` → 缺数据文件，指向 `transfer`；
3. `ImportError: cannot import name '<x>' from '<y>'` → 入口 import 写错；
4. `*** JOB <id> ON <节点> CANCELLED ... DUE TO TIME LIMIT ***` → 撞墙钟限制
   （匹配器前缀无关：兼容 107 上实测的 `[ISO-时间戳] error:` 新格式，不带
   `slurmstepd:` 前缀）；
5. `Detected N oom-kill event(s)` → Slurm OOM 击杀；
6. `torch.OutOfMemoryError: CUDA out of memory...` → **解析**申请量/剩余量/
   总量（如「申请 40 GiB 但仅 30.86 GiB 空闲」），给出针对性修法；
7. 兜底 `XxxError/XxxException: 信息` → 引用 traceback。

证据收集（`cluster/monitor.py`）：`sacct_row` 用 `sacct -j <id> -X --parsable2`
读 State/ExitCode；`collect_logs` 同时读 `.out` 和 `.err`（traceback 与
slurmstepd 错误落在 `.err`——这是实机采集发现的），截断到 64 KiB。
降级路径：作业号不在账本 → 警告；日志为空 → 只看退出码；`0:0` 无签名 →
报告「未发现失败」；无签名且非零 → 打印日志末 5 行供人工判断。

`diagnose()` 本身是纯函数（无 I/O），因此整个匹配库可离线单测。

## 9. 校验（core/validate.py）

- **注入安全**：entry/partition/qos/account/mem/time/log_dir/conda_env/
  job_name 拒绝空白与 shell 元字符（`;&|<>$` 等）——这些值会被写进 sbatch。
- **范围**：entry 非空；gpu ≥ 0；cpus/nodes/ntasks ≥ 1。
- **array 语法**：正则校验 `N`、`N-M`、`N-M:S`、逗号列表、可选 `%K` 并发上限
  （`:0`、`%0` 拒绝）。
- **多节点 DDP**：`ntasks == nodes`（每个节点恰好一个 torchrun 启动器）。

## 10. 工作流状态（core/workflow.py）

`workflow_checks` 区分两类信号：

- **errors（阻塞）**：`conda_env` 已配置但环境从未创建——渲染出的
  `conda activate` 会立刻杀死作业，属确定性失败，`submit` 直接中止
  （`--skip-check` 可绕过）。
- **warnings（只提示）**：环境存在但 `conda_env` 未接线；无环境且检测到
  第三方依赖（列出样例 import）；无环境且无依赖。
- conda 不可用时**开闸失败**（fail open）：什么都不查，不阻塞。

`workflow_status` 把同一状态渲染成 `lazy107 check` 的仪表盘：入口、环境、
记录数、命令史条数，末尾一条唯一的 `next:` 建议
（`init` / `env --yes` / `submit` / `watch <最新作业>`）。

## 11. 记账（core/record.py）

- `notes/runs.md`：每次成功提交追加 `## Job <id>` 段（时间戳、批处理脚本
  路径、入口、account/array（若有）、分区、QoS）。同时兼容旧格式 `## Run`。
- `notes/history.md`：项目目录内每执行一条 lazy107 命令追加一行
  `- [时间戳] lazy107 ... (exit N)`。写入失败静默吞掉——记账永远不破坏命令。
  判定「在项目内」的门槛是存在 `107.toml`、`notes/` 或脚手架自带的
  `environment.yml`，避免从 `$HOME` 误记账。

## 12. 集群 I/O 层

### discover（cluster/discover.py）

- `sacctmgr show assoc user=<u> format=Account,Partition,QOS --parsable2
  --noheader`：机器格式解析，杜绝列截断；`scontrol show partition` 用正则
  在整行里找 `AllowAccounts/AllowQos/Default/MaxTime/MaxMemPerNode`
  （这些字段被 Slurm 打包在同一行，前缀匹配会漏）。
- `resolve_association` 确定性打分：默认账户 +4、默认分区 +2、显式关联分区
  +1，同分再比分区/行序；QoS 取第一个「已分配且被允许」的。结果写入
  `~/.config/lazy107/config.toml`（唯一写操作，只写用户自己的文件）。
- `submission_check` 是提交时的守卫，与 discover 共享同一份解析，但**只凭
  正面证据拦截**：account 不在 AllowAccounts、QoS 未分配给账户、QoS 不在
  AllowQos、`time` 超过分区 MaxTime、`mem` 超过 MaxMemPerNode（后两项仅
  双方都可解析时才比较）。Slurm 数据缺失时开闸放行（`sinfo` 预检仍兜底）。

### env（cluster/env.py）

- conda 定位：PATH 优先，其次 `~/miniconda3/bin/conda`。
- 命令序列确定性、可打印：

  ```bash
  module load miniconda/py312    # conda 已在 PATH 时为空操作
  conda create -y -n <env> python=3.12
  conda env update -n <env> -f environment.yml      # 有则执行
  conda run -n <env> pip install -r requirements.txt # 有则执行
  # 或（仅 pyproject.toml）：conda run -n <env> pip install -e .
  # GPU 作业追加 pip 覆盖（见 §14）
  ```

- 环境已存在则跳过 create（其余步骤幂等）。
- **环境复用校验** `missing_deps`：一次 `conda run` 对全部依赖做
  `importlib.util.find_spec` 探测（不执行 import）；探测失败**关闸**
  （全部报缺失）；无法映射 import 名的包跳过不误报。
- `gpu_build_check` 三态：`None`（torch 未装）/`""`（CPU-only 构建）/
  `"12.6"`（CUDA 版本）。特意不用 `torch.cuda.is_available()`——它在登录
  节点恒为 False，证明不了任何东西。

### monitor / submit（cluster/monitor.py、cluster/submit.py）

- `monitor_commands` 生成 4 条监控命令（scontrol show job / squeue --me /
  tail -f .out / tail -f .err）；数组作业时 tail 目标换成通配集。
- `submit_sbatch` 调用 `sbatch`，解析标准回复 `Submitted batch job <id>`；
  失败抛出带 stderr 的 `SubmitError`（不打印 traceback）。
- `slurm_check` 用 `sinfo -p <分区> -h` 预检分区。

## 13. 提交流水线（cmd_submit 的完整顺序）

顺序即设计——每一阶段在最后一步前都是只读的：

1. **解析**：entry 参数 > 107.toml 固定值 > 自动检测；五层配置合并。
2. **校验**：`validate_plan` + `detect_ddp` 约束（§9）。
3. **Notebook**：入口是 `.ipynb` → 就地转换 `.py`（nbconvert 懒加载，
   未装则从 JSON 直接抽代码单元格；剥掉 IPython/ipywidgets/display/plt.show
   等交互代码；旧文件先备份）。
4. **工作流检查**：§10 的 errors/warnings。
5. **Slurm 预检**：`sinfo` 分区存在性 → `submission_check` 关联与上限校验。
6. **GPU 预检**：`gpu>0` 时校验环境 torch 的 CUDA 构建，CPU-only 即拦截。
7. **渲染**：`--dry-run` 在此短路——打印脚本，不写文件、不提交。
8. **确认**：无 `--yes` 时问 `submit ... ? [y/N]`；回答 `n`（或非交互
   shell 的 EOF）→ 中止且**零产物**。
9. **写脚本**：`scripts/<入口>.sbatch`（旧文件备份）。
10. **提交**：`sbatch`，捕获作业号。
11. **记账**：追加 `notes/runs.md`。
12. **汇报**：`submitted job <id> (<作业名>)` + `watch: lazy107 watch <id>`。

## 14. GPU/CUDA 完整性（四道防线）

不变式：**「申请了 GPU」必须等于「CUDA」，绝不允许静默退化为 CPU 训练。**
四个独立执行点，全部有测试覆盖：

1. **安装时锁定**（`env.py::_gpu_overrides`）：GPU 计划且检测到 GPU 框架时，
   最后的安装命令必是 pip 默认源——`torch torchvision`（默认 wheel 自带
   CUDA）、`jax[cuda12]`（默认 jax 是 CPU-only）、`tensorflow`（默认即 GPU）。
   conda 频道的 CPU torch 永远不可能成为最终安装。
2. **装后校验**（`gpu_build_check`）：安装完毕查询 `torch.version.cuda`，
   三态判断（§12），CPU-only 或未装即失败。
3. **提交前拦截**（`gpu_preflight_error`）：`gpu>0` + CPU-only torch →
   提交中止并给出重装修法。
4. **运行时断言**（`render.py::_GPU_GUARD`）：`gpu>0` 时脚本内置一段
   `importlib.util.find_spec('torch')` 守卫——有 torch 才断言
   `torch.cuda.is_available()`（纯 CPU 项目不受影响）。作业环境里 CUDA
   缺失时响亮失败，而不是默默跑完一个 CPU 训练。

## 15. everything 向导（wizard.py）

`everything` 是**外部脚手架**：懒加载（`build_parser` 内 import，避免循环
依赖），内部只编排既有 `cmd_*` 函数（用 `SimpleNamespace` 传参）与
`cli.py` 的交互辅助——**零逻辑重复**。流程：

```text
1. 入口选择（多候选 + TTY 时；选择固定进 107.toml）
2. 环境：conda_env 已接线 → 跳过；
         否则列出已有环境供复用（校验通过即固定）；
         没有则询问是否新装（调 cmd_env --yes）
3. Slurm 资源：5 档预设或 c 逐字段编辑
4. 打印 plan（顺带完整校验，失败即中止）
5. 可选预览 sbatch（--dry-run）
6. 提交（cmd_submit 自己的确认提示是最后一道闸）
```

预设表（`PRESETS`）：

| 预设 | gpu | cpus | mem | time |
|---|---|---|---|---|
| `cpu-light` | 0 | 2 | 4G | 1:00:00 |
| `cpu-heavy` | 0 | 8 | 32G | 8:00:00 |
| `gpu-light` | 1 | 4 | 16G | 2:00:00 |
| `gpu-heavy` | 1 | 8 | 64G | 12:00:00 |
| `gpu-multi` | 2 | 16 | 64G | 24:00:00 |

- 回车 = 保持当前值；选预设 = 一次写入四个键（与当前值相同的键会被丢弃，
  不做无意义改写）；`c` = 逐字段提示（带常见值提示）。
- gpu/cpus 强制整数（非法则重问）；mem/time 原样透传；范围与安全校验
  刻意推迟到随后的 `validate_plan`。
- `--yes` 非交互：自动入口检测、全新装环境、保留当前资源、直接提交。

设计立场（诚实预设）：预设是**起点**而不是预测。真实的测量闭环在平台里——
`debug` 读出 `0:9` → 提高 `mem`；`0:15` → 提高 `time`——老手手工也是这么
干的，向导只是让第一次选择变得便宜。

## 16. 模板与传输（template.py、transfer.py）

- 模板作为包内数据随 wheel 分发（`importlib.resources`），`init` 由此
  脚手架 9 个文件：`src/{data,model,train}.py`、`src/__init__.py`、
  `environment.yml`、`pyproject.toml`（依赖含 torch）、`scripts/train.sbatch`、
  `.gitignore`、`README.md`；`{project}` 占位符替换为项目名；
  拒绝写入非空目录。
- `transfer` 用一组后缀（`.pt .pth .ckpt .safetensors .onnx .h5 .hdf5 .npz
  .npy .tar .zip .gz .bin .db .sqlite .parquet`）与目录名（`data datasets
  checkpoints outputs logs models`）发现大文件，打印
  「打包 → GUI 上传 → 解包 → sha256 校验」清单；清单永不包含自己的归档
  文件；只打印，从不执行（模板 `.gitignore` 与这组规则由测试强制同步）。

## 17. 错误处理与安全边界

### 错误处理

- 所有用户可见失败在 `main()` 捕获（`ValueError`/`FileNotFoundError`/
  `FileExistsError`/`SubmitError`），以 `lazy107: <信息>` 打印到 stderr、
  退出码 1——出现 traceback 意味着这是 bug 而不是用户错误。
- `sbatch` 失败带 stderr 原文；`env` 的安装命令带超时（每条 15 分钟）；
  conda 探测带 30 秒超时。
- `env --dry-run` 在没有 conda 的机器上也能跑——打印命令是纯输出。

### 安全边界

- **登录节点只做管理**：训练永远走 `sbatch`。
- **`--dry-run` 只读**：不写文件、不提交（`render` 与 `submit` 都验证过）。
- **`transfer` 只打印**：不自动同步、不删除任何东西。
- **确认提示被尊重**：`n`/EOF → 中止且零产物。
- **GPU 即 CUDA**：§14 四道防线。
- **文件里没有秘密**：配置/脚本/日志/账本都不存放 token 或凭据。
- **标识符校验**：所有会进入 shell 的自由字段拒绝元字符；sbatch 路径
  始终项目相对。

## 18. 平台事实（USTC 107，实测）

- 分区：`P107-RTX5090`（**默认**，15×RTX5090 节点 / 120 卡）、`P107-A100`
  （11×A100，MaxNodes=2）、`GPU-RTX5090`、`GPU-A100`、`CPU-6530`、
  `CPU-8358P`、`Students`（26 节点）。
- 访问控制：`P107-*` 仅允许账户 `competition`（QoS `qos_p107-rtx5090` /
  `qos_p107-a100`）；`Students` 允许 `stu,stu001,demo_admin,cmet`
  （`qos_stu_*` 族）。竞赛账户提交到 `Students` 必被拒——用 `discover`。
- QoS 限额：`qos_p107-*`：MaxWall=4d，MaxTRESPU cpu=16 gpu=4，MaxJobsPU=4。
- 模块：`miniconda/py312`、`python3.12`、`cuda/12.6`、`cuda/13.0`、`apptainer`。
- pip 镜像：`https://mirrors.ustc.edu.cn/pypi/web/simple`。
- conda 批处理激活：`source "$(conda info --base)/etc/profile.d/conda.sh"`
  再 `conda activate`（绝不 `source activate`）。
- 日志事实（2026-09-03 实机采集，见 `examples/failures/SIGNATURES.md`）：
  失败信息几乎全部落在 `.err`（`.out` 常为空）；超时行是
  `[ISO-时间戳] error:` 前缀（新 Slurm 格式，无 `slurmstepd:`）；**P107-RTX5090
  上 `--mem` 实测未强制执行**（4G 请求的作业活到超时被杀，未触发 OOM 击杀）。

## 19. 测试

`tests/` 是行为规格：19 个文件、274 个用例，全部离线可跑（零网络依赖、
子进程均被 fake）。分层规则使 `core/*` 的纯逻辑可以不带集群直接验证；
`cluster/*` 通过注入/fake 覆盖。运行：

```bash
python -m pytest        # pyproject 已配置 addopts --basetemp=.pytest-tmp
```

## 20. 端到端数据流（一图总结）

```text
项目目录(.py/.ipynb/依赖文件)
   │  依赖推断(§5) + 入口检测(§6)
   ▼
RunPlan(§3) ──配置合并(§4)── 校验(§9)
   │
   ├─ env --yes ──► conda 环境(§12) ──► CUDA 校验(§14.2) ──► 107.toml 写回(§4)
   ├─ render ────► 确定性 sbatch(§7) ──► GPU 守卫(§14.4)
   └─ submit ────► 流水线(§13) ──► sbatch ──► notes/runs.md(§11)
                         │
                   watch/logs ──► 日志
                   debug ───────► 诊断(§8)
```
