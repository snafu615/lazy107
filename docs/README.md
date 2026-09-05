# lazy107 文档

lazy107 是面向 USTC 107 计算平台的集群端作业提交命令行工具。
新用户从[快速开始](quickstart.md)读起；想理解内部实现看[设计文档](design.md)。

## 文档索引

| 文档 | 读者 | 内容 |
|---|---|---|
| [intro.md](../intro.md) | 评委 | 作品简介（提交材料）：背景、问题、功能、模型/API 说明、创新点 |
| [quickstart.md](quickstart.md) | 新用户 | 从安装到第一个提交作业的分步教程 |
| [demo.md](demo.md) | 所有人 | 端到端完整实例：ResNet-18/CIFAR-10 真实训练（逐步） |
| [design.md](design.md) | 开发者、评委 | 架构、代码结构、各模块职责、GPU/CUDA 完整性、安全边界 |
| [demos/INDEX.md](demos/INDEX.md) | 评委、新用户 | 六个实机演示（安装探测、happy path、向导、debug、自适应渲染、防护栏杆）|
| [runbook.md](runbook.md) | 验收测试 | 分步验收清单与常见阻塞 |
| [env-pipeline.md](env-pipeline.md) | 开发者 | 环境流水线详解：分析 → 生成 → 执行 → 消费 |

## 语言说明

主文档（本页、quickstart、demo、design）为中文；`demos/`、`runbook.md`、
`env-pipeline.md` 暂为英文，翻译陆续跟进。平台事实（分区、QoS、模块、
镜像）均与 USTC 107 训练材料及平台 Slurm 脚本规范交叉核对过。
