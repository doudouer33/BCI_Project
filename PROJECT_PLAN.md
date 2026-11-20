# 数据挖掘 Project 3：EEG 运动想象分类项目实施规划

> **项目名称**：基于机器学习与 Transformer 的 EEG 表征学习与运动想象分类  
> **课程方向**：Project 3 — AI4Science：数字孪生脑与脑机接口（BCI）  
> **小组成员**：邓泽林、许亦然、万志伟  
> **计划完成时间**：2026 年 11 月 30 日  
> **文档状态**：实施规划（不代表代码、实验或模型已完成）

## 1. 项目定位与研究目标

本项目延续小组 Proposal 和中期汇报确定的技术路线，在公开运动想象脑电（EEG）数据集上开展**四分类**任务，对比传统人工特征与深度表征学习方法。项目的重点不是构建完整的“数字孪生脑”，也不是复现老师提供的所有 BCI 论文，而是在一致、可复现的实验协议下回答以下问题：

1. CSP 这类传统空间特征工程在 EEG 运动想象分类中能达到怎样的效果？
2. EEGNet 等紧凑型深度模型能否相较 CSP + LDA/SVM 获得稳定收益？
3. 若加入轻量 Transformer，额外的模型复杂度是否带来值得关注的改进？
4. 模型效果在不同受试者、不同运动想象类别之间有何差异？

**基本原则**：先完成真实、可信的基线结果，再推进深度学习与进阶模型；先锁定数据及评估协议，再进行调参；所有结论以实际运行结果为依据。

### 1.1 工作范围与优先级

| 优先级 | 工作内容 | 最终要求 |
| --- | --- | --- |
| **P0：必须完成** | BCI IV 2a 读取、预处理与可视化 | 四分类、9 名受试者、可复现数据流程 |
| **P0：必须完成** | CSP + LDA | 9 名受试者的真实测试结果 |
| **P0：必须完成** | CSP + SVM | 9 名受试者的真实测试结果 |
| **P0：必须完成** | EEGNet | 9 名受试者的真实测试结果与训练记录 |
| **P0：必须完成** | 模型统一评价、分析和 Final Report | 指标表、混淆矩阵、训练曲线、误差分析、复现说明 |
| **P1：争取完成** | 轻量 Transformer / EEG-Deformer 风格模型 | 至少一组按同样协议开展的实验；尽量覆盖 9 人 |
| **P1：争取完成** | 一组参数敏感性或对照实验 | 改动单一因素并解释观察到的差异 |
| **P2：文献讨论为主** | LaBraM、EEGPT、CBraMod 等 EEG 基础模型 | 相关工作与未来方向；不要求完整预训练/迁移复现 |

> 如果时间不足，保留 P0 的完整性，削减 P1/P2。**不得用计划结果或论文中的数字代替自己实际跑出的结果。**

## 2. 数据集与任务定义

### 2.1 数据集

- **名称**：BCI Competition IV — Dataset 2a（MOABB 中为 `BNCI2014_001`）。
- **被试**：9 名。
- **EEG 通道**：22 个（另有非 EEG 通道，读取时需确认只使用约定的 EEG 输入）。
- **采样率**：原始数据为 250 Hz；若重采样，应统一并记录。
- **类别**：左手（Left Hand）、右手（Right Hand）、双脚（Feet）、舌头（Tongue）运动想象，**保持 4 类，不降成 2 类**。
- **Session**：每名受试者有两次记录；官方设计中每次 288 个试次，理论总量为 `9 × 2 × 288 = 5184` 个试次。最终实际有效样本数以程序读取、检查后的结果为准。

数据入口：

- [MOABB：BNCI2014_001](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)
- [MOABB：MotorImagery](https://moabb.neurotechx.com/docs/generated/moabb.paradigms.MotorImagery.html)
- [MOABB 运动想象入门示例](https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_1_simple_example_motor_imagery.html)

### 2.2 统一输入与标签

数据处理模块向所有模型提供一致的：

- `X`：三维 EEG 数组，形状为 `(n_trials, n_channels, n_times)`；
- `y`：四分类标签，推荐固定编码 `0/1/2/3`，保存**明确的类名↔整数映射**；
- `metadata`：至少包含 `subject`、`session`、`run`、试次标识及必要的标签信息。

**不要将 `(576, 22, 1001)` 之类的示例形状写死在代码里**：时间点数取决于事件对齐方式、截取时间窗、采样率及库的端点约定。应在运行时检查并打印维度。

## 3. 统一实验协议（开始训练前锁定）

这是项目中最需要保持一致的部分。所有模型共用同一数据源、相同四分类标签、相同测试 Session 和相同评价方式。

| 项目 | 第一版统一约定 |
| --- | --- |
| 评估任务 | **同一受试者跨 Session 四分类**；每位受试者单独训练、单独测试 |
| 训练/测试 | 使用训练 Session（例如 MOABB 的 `0train`）开发模型，测试 Session（例如 `1test`）仅用于最终测试；**实际字符串必须先检查** |
| 验证集 | 只从训练 Session 中划出；建议固定随机种子的分层划分（如 80% 训练、20% 验证），保存索引，所有模型复用 |
| 滤波 | 第一版统一使用 **8–30 Hz** 带通滤波；记录处理库、参数及滤波实现 |
| 时间窗 | 建议使用**相对于运动想象提示（cue）开始后 0–4 s** 的候选窗口；先核对数据集/MOABB 的事件时间定义，再将实际采用的 `tmin/tmax` 锁定到配置文件 |
| 采样率 | 第一版优先保留原采样率；如重采样，所有模型一致并记录 |
| 标准化 | 需要估计均值/方差等统计量时，**仅在训练集拟合**；验证/测试只执行 `transform`。CSP 及分类器参数也必须在训练数据上拟合 |
| 随机性 | 固定随机种子，保存 Python 和核心依赖版本、训练配置 |
| 测试纪律 | 禁止根据测试 Session 表现反复选择时间窗、滤波参数、超参数、随机种子或检查点 |

**关于模型公平性**：共享相同的 EEG 样本和最终测试集，但允许模型使用必要、明确记录的模型专属处理（例如 SVM 的特征缩放、EEGNet 的输入数值缩放）。这些处理只能从训练数据中学习，不能利用测试集统计量。验证集用于调参与早停；最终结果必须记录所采用的训练策略。

### 3.1 防止数据泄漏的明确规则

1. **先确定 train/val/test 索引，再拟合任何需要学习的变换。**
2. CSP 只能使用训练部分拟合空间滤波器；应将其放在 `scikit-learn Pipeline` 中与分类器一起训练。
3. `StandardScaler` 等预处理器只使用训练部分 `fit`，在验证/测试部分只 `transform`。
4. EEGNet、Transformer 的 `early stopping` 和超参数选择只看验证集，不看测试集。
5. 不能把两个 Session 随机混合后重新划分训练/测试；也不能把一个被试的最好成绩当作项目平均成绩。
6. 滤波、事件对齐、标签顺序、通道选择及单位（V/μV）必须记录并复核。

## 4. 技术路线与模型设计

```text
BCI Competition IV 2a (BNCI2014_001)
                │
                ▼
        MOABB / MNE 读取
                │
                ▼
      EEG 通道选择与预处理
    (8–30 Hz / Epoch / 标签)
                │
                ▼
      按受试者及 Session 划分
     train / validation / test
                │
       ┌────────┼──────────┐
       ▼        ▼          ▼
  CSP 特征     EEGNet   轻量 Transformer
   ├── LDA      CNN        (进阶可选)
   └── SVM
       └────────┼──────────┘
                ▼
     9 名受试者统一指标汇总
   Accuracy / Macro-F1 / 混淆矩阵
                │
                ▼
      模型对比、误差分析、报告
```

### 4.1 传统基线：CSP + LDA / CSP + SVM

- **CSP**：提取与运动想象类别有关的空间方差特征；实现中确认使用的是**多分类** CSP。
- **LDA**：优先跑通，用作最简、可解释的 baseline。
- **SVM**：共享 CSP 特征流程，在训练/验证集选择 `C`、`kernel` 等参数。
- CSP 初始特征维数可设置为 `n_components=8`，随后在训练 Session 内部验证是否调整。
- 输出每位受试者的 Accuracy、Macro-F1、混淆矩阵、预测标签和超参数。

推荐组件：`mne.decoding.CSP`、`sklearn.pipeline.Pipeline`、`LinearDiscriminantAnalysis`、`SVC`。

### 4.2 核心深度模型：EEGNet

- 使用 **PyTorch + Braindecode** 提供的 EEGNet 实现，避免不必要地重写网络细节。
- 输入 `(batch, channels, timepoints)`，输出 `(batch, 4)` 的分类 logits；实际张量形状以当前模型 API 为准并在训练前验证。
- 初始训练配置：Adam、交叉熵损失、固定随机种子、训练集内部验证、early stopping。
- 保存训练/验证损失曲线、验证集最佳轮次、最终测试指标以及必要的模型检查点。
- 先跑 Subject 1，确认能正常前向、反向传播，再扩展到所有 9 名受试者。

官方参考：[Braindecode EEGNet API](https://braindecode.org/stable/generated/braindecode.models.EEGNet.html)。具体初始化参数以安装版本的 API 为准。

### 4.3 进阶模型：轻量 Transformer / EEG-Deformer 风格

**仅在 P0 的完整模型对比已建立后开始。**

- 首选低复杂度路线：EEG 时间窗切块/embedding → 少量 Transformer Encoder → 池化或分类 token → 四分类头。
- 可借鉴老师列出的 **EEG-Deformer** 的卷积与 Transformer 结合思路，但不要求逐细节完整复现。
- 与前三个模型使用相同数据切分与最终评价指标；记录参数量、训练耗时（若条件允许）。
- 即使准确率不及 EEGNet，也应如实报告并分析可能的过拟合、数据量或模型设计因素。

LaBraM、EEGPT、CBraMod 等大规模预训练模型作为文献综述和未来工作的参考；**不把大规模预训练设为最低交付要求**。

## 5. 项目目录与文件约定

项目根目录建议命名为 `BCI_Project/`。采用**一套共享数据接口、按模型分训练脚本、结果集中管理**的结构；先保证必要文件，避免为了课程项目过度工程化。

```text
BCI_Project/
├── README.md                      # 项目介绍、安装和复现命令
├── PROJECT_PLAN.md                # 本文：整体规划与验收清单
├── requirements.txt               # 固定依赖版本（运行后导出）
├── .gitignore                     # 忽略数据集缓存、模型权重、环境目录等
│
├── configs/
│   └── experiment.yaml            # 统一的滤波、时间窗、划分、随机种子及模型参数
│
├── data/
│   ├── raw/                       # 原始 EEG 文件或缓存（不提交 Git）
│   └── processed/                 # 可选的预处理缓存与数据统计（不提交大文件）
│
├── src/
│   ├── load_data.py               # MOABB 读取、预处理、标签编码、Session 划分
│   ├── train_csp.py               # CSP + LDA / SVM 基线训练与测试
│   ├── train_eegnet.py            # EEGNet 训练、验证、测试
│   ├── train_transformer.py       # 可选：轻量 Transformer
│   ├── evaluate.py                # Accuracy、Macro-F1、混淆矩阵与 CSV 汇总
│   └── plot_results.py            # 数据波形、频谱、对比图与训练曲线
│
├── results/
│   ├── metrics.csv                # 所有模型逐受试者的最终指标
│   ├── predictions/               # 测试标签及预测结果，支持复核
│   ├── figures/                   # EEG 图、准确率对比、混淆矩阵、损失曲线
│   ├── logs/                      # 训练日志、超参数、运行配置
│   └── models/                    # 可选：最佳模型权重（大文件不提交 Git）
│
├── presentation/
│   └── midterm/                   # 已有的中期汇报材料（如实际存在）
│
└── report/
    ├── figures/                   # 可选：最终报告所用插图
    └── final_report.pdf           # 最终实验报告
```

**注意**：上面是目标目录结构，不表示这些脚本和结果文件已经存在。可根据实际编码合并少量脚本，但应保持职责清晰。

### 5.1 统一数据接口（建议）

所有模型只调用 `load_data.py` 提供的共享接口，而不是三个人各写一套预处理。

```python
# 示意接口：实际实现时确定签名并写入 README
X, y, metadata = load_subject(subject_id=1, config=config)
train_idx, val_idx, test_idx = get_split_indices(metadata, config)
```

接口要求：

- 载入任意一个受试者，不在函数内部混入其他受试者；
- 返回的通道、采样率、标签顺序、时间窗在各模型间一致；
- 保存训练/验证/测试索引或可严格复现的划分规则；
- 输出检查日志：`X.shape`、四类数量、各 Session 数量、NaN/Inf 数量、信号单位；
- 原始数据下载位置可配置，避免仓库中硬编码个人绝对路径。

### 5.2 配置文件的最小字段（示意，不代表已生效）

```yaml
dataset: BNCI2014_001
subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
n_classes: 4
fmin: 8
fmax: 30
# 对齐 cue 的具体定义必须先与 MOABB 实际事件核对
tmin: 0.0
tmax: 4.0
resample_sfreq: null
train_session: 0train
test_session: 1test
val_ratio: 0.2
random_seed: 42
csp_components: 8
```

首次执行时，确认 MOABB 实际的 `session` 名称与 `tmin/tmax` 对齐定义，之后再固定配置并生成实验结果。

### 5.3 结果文件统一格式

`results/metrics.csv` 至少包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `subject` | 受试者编号（1–9） |
| `model` | `CSP_LDA` / `CSP_SVM` / `EEGNet` / `Transformer` |
| `train_session`、`test_session` | 本次实验的 Session 来源 |
| `accuracy` | 该受试者测试集准确率 |
| `macro_f1` | 该受试者测试集 Macro-F1 |
| `seed` | 随机种子 |
| `config_id` | 对应的实验参数配置编号或路径 |

建议同时保存：

- `results/predictions/<model>_s<subject>.csv`：`trial_id,y_true,y_pred`；
- `results/logs/`：训练参数、最佳验证轮次、损失曲线原始数据；
- `results/figures/`：由程序重新生成的正式图片；
- `results/models/`：如需要复现深度模型推理，可保存权重。

最终汇总使用**9 名受试者指标的均值**，最好同时给出标准差；不要只报告最好的一名受试者。若报告总体混淆矩阵，注明是逐人矩阵求和还是其他统计口径。

## 6. 分阶段实施与验收

### 阶段 A：数据读取、预处理和可视化（P0）

**实施**

1. 搭建 Python 环境，安装并记录 MOABB、MNE、NumPy、scikit-learn、PyTorch、Braindecode 等版本。
2. 用 MOABB 读取 Subject 1，核对 EEG 通道、采样率、四类标签、Session、Run 及样本数。
3. 实现统一的 8–30 Hz 滤波、事件分段、时间窗选择和样本/标签接口。
4. 生成至少三种**实际数据图**：EEG 时域波形、四类样本分布、功率谱/频段差异。
5. 读通 9 名受试者，保存数据统计，并与其余两名成员确认接口。

**验收**：`load_data.py` 可重复运行；四类正确；9 人数据统计完整；无未解释的维度、单位或标签异常；至少 3 张真实图，带坐标轴、单位和图注。

### 阶段 B：CSP + LDA/SVM 基线（P0）

**实施**

1. 先在 Subject 1 跑通 CSP + LDA，检查训练/验证/测试划分及评价脚本。
2. 复用 CSP 数据流程，加入 CSP + SVM。
3. 在训练 Session 内验证 CSP 维数及 SVM 参数；**不根据测试成绩调参**。
4. 扩展到 9 名受试者，导出逐人 Accuracy、Macro-F1、预测标签与混淆矩阵。

**验收**：两个传统模型各有 9 人结果；能从配置与脚本复现；结果没有明显的数据泄漏迹象。

### 阶段 C：EEGNet（P0）

**实施**

1. 验证一个 batch 的输入形状、标签类型及输出 `(batch, 4)`。
2. 用训练 Session 中的训练/验证部分调试学习率、batch size、early stopping。
3. 保存 Subject 1 的训练/验证曲线，确认训练稳定后扩展到其余 8 人。
4. 使用与 CSP 基线一致的测试 Session，汇总全部 9 人的指标与混淆矩阵。

**验收**：9 人 EEGNet 真实结果；至少一张训练/验证损失曲线；实验配置、随机种子和必要训练日志齐全。

### 阶段 D：轻量 Transformer（P1，选做）

**实施**

1. 在已有 EEGNet 输入管线基础上设计轻量 patch/embedding + Encoder + 分类头。
2. 先完成单人完整训练及测试，再视资源扩展到 9 人。
3. 记录模型参数量和训练耗时（条件允许时）；对比 EEGNet 的收益与代价。

**验收**：只把真实完成的部分写入报告；未完成 9 人实验时，必须明确标注覆盖范围，不能与完整 9 人均值混为一谈。

### 阶段 E：对照实验、结果分析和 Final Report（P0 + P1）

**实施**

1. 统一汇总 9 人 × 3 个必做模型的 Accuracy、Macro-F1，绘制对比图。
2. 制作代表性混淆矩阵，分析左右手/双脚/舌头哪些类别容易混淆。
3. 分析受试者差异，检查深度模型是否**稳定**优于传统模型。
4. 至少争取开展一组简单的参数敏感性实验，如 `CSP n_components = 4 / 8 / 12`；固定其他因素，用训练/验证数据确定设置，避免反复利用测试集选择结论。
5. 编写 Final Report，完整整理方法、指标、实验结果、讨论、局限及参考文献。

**验收**：报告图表均能追溯到实际日志/CSV；结果口径一致；有方法比较和局限讨论，而不是只罗列准确率。

## 7. 小组协作与责任划分

沿用中期汇报中提出的分工；具体任务可由组内协商调整，但应保持**一人维护共享数据处理协议、多人共用同一评价接口**。

| 成员 | 主要负责 | 关键交付 |
| --- | --- | --- |
| 邓泽林 | 数据读取、预处理、可视化与统一数据接口；整合数据与报告材料 | `load_data.py`、数据统计、EEG 波形/频谱/类别分布图、实验配置 |
| 许亦然 | 传统机器学习与基线分析 | `train_csp.py`、LDA/SVM 9 人结果、参数记录、基线对比 |
| 万志伟 | EEGNet 与可选 Transformer | `train_eegnet.py`、可选 `train_transformer.py`、9 人深度模型结果、训练曲线 |
| 三人共同 | 协议复核、评价分析、Final Report 与材料归档 | `metrics.csv`、模型对比、误差分析、最终 PDF、README |

**协作约定**：

- 数据处理的事件对齐、标签映射、时间窗、Session 划分由小组确认一次后统一使用。
- 模型训练代码不重复实现私有数据读取规则；需要特殊处理时写入配置与日志。
- 新结果先核对 `subject / model / split / config` 再写入统一 `metrics.csv`。
- 每个实验至少保留运行命令、配置、依赖版本、随机种子、预测输出及指标。
- 中期材料可作为 Final 的背景和前期规划依据；但不能把中期方案中的“待实测”当作已取得的结果。

## 8. 剩余时间安排（2026-11-20 至 2026-11-30）

> 下表是从**当前规划开始执行**的建议节奏，不代表此前阶段已经完成。若某项实际已完成，直接跳到核验与下一项即可。

| 日期 | 主要工作 | 当日/阶段目标 |
| --- | --- | --- |
| **11/20–11/21** | 环境与数据、Subject 1 读取；固定实验协议；扩展 9 人、生成 3 张数据图 | `load_data.py` 与统一配置可运行 |
| **11/22–11/23** | CSP + LDA、CSP + SVM：单人调通并完成 9 人实验 | 两个基线的逐人指标、预测输出 |
| **11/24–11/26** | EEGNet：Subject 1 调试，再完成 9 人训练/测试 | EEGNet 逐人指标、训练曲线与日志 |
| **11/27** | 核对三模型结果；若 P0 已完整，尝试轻量 Transformer | P0 对比表完整；可选 Transformer 初步结果 |
| **11/28** | 混淆矩阵、对比图、受试者分析；争取一组参数敏感性实验 | Final Report 所需核心图表 |
| **11/29** | 报告撰写、引用与成员贡献整理、README 复现说明 | 最终报告初稿、代码与结果归档 |
| **11/30** | 核对数据/图表/结论一致性，完成 PDF 与最终提交 | 提交可复现的完整成果 |

### 停损规则

- 如果数据仍无法完整读取，**暂停 Transformer 和消融实验**，先确保 Subject 1 → 9 人数据和 CSP + LDA 跑通。
- 如果 EEGNet 尚不能稳定训练，优先检查标签映射、数值单位、输入维度、损失函数及学习率，不增加新模型。
- 到 **11/27** 如果仍没有可靠的 Transformer 结果，停止扩展进阶模型，将精力转向前三模型的复核、分析和报告。
- 截止前若时间不足，以**9 人 × 3 个必做模型的可信结果 + 清楚的实验分析 + 复现材料**为第一交付目标。

## 9. 统一评价与 Final Report 结构

### 9.1 评价指标

- **Accuracy**：分类正确试次 / 测试总试次。
- **Macro-F1**：对四个类别分别计算 F1 后求宏平均；与 Accuracy 一起报告。
- **Confusion Matrix**：展示具体类别混淆，不只报告单个数字。
- **逐受试者统计**：每个模型分别列出 9 名受试者的测试结果，再汇总均值（建议加标准差）。
- **可选复杂度比较**：参数量、每受试者训练耗时；需要注明计算硬件和统计口径。

建议结果表（**数值待真实运行后填写**）：

| 模型 | 9 人平均 Accuracy | 9 人平均 Macro-F1 | 逐人结果 | 备注 |
| --- | --- | --- | --- | --- |
| CSP + LDA | 待实测 | 待实测 | 待实测 | P0 |
| CSP + SVM | 待实测 | 待实测 | 待实测 | P0 |
| EEGNet | 待实测 | 待实测 | 待实测 | P0 |
| 轻量 Transformer | 待实测 | 待实测 | 待实测 | P1 / 按实际完成范围报告 |

### 9.2 Final Report 建议大纲

1. **Introduction**：BCI/EEG 背景、运动想象四分类任务、研究问题与贡献。
2. **Related Work**：CSP、EEGNet、EEG-Deformer / Transformer、LaBraM / EEGPT / CBraMod 等相关工作。
3. **Dataset & Preprocessing**：BCI IV 2a 数据描述、滤波、时间窗、标签、可视化与 Session 划分。
4. **Methods**：CSP + LDA/SVM、EEGNet，以及实际完成的 Transformer 方法。
5. **Experiments & Results**：统一评价协议、超参数、9 人结果表、混淆矩阵、训练曲线与可选对照实验。
6. **Discussion**：性能差异、不同受试者与类别的困难、模型复杂度、失败原因与局限性。
7. **Conclusion & Future Work**：基于真实结果的结论；跨被试泛化、预训练迁移等未来方向。
8. **Contributions & References**：三名成员实际工作与引用文献。

### 9.3 需要保留的关键图表

- EEG 时域波形图；
- 四类样本分布图；
- EEG 功率谱图；
- 9 人 × 3 个主要模型的 Accuracy 对比图；
- 代表性混淆矩阵；
- EEGNet 训练/验证曲线；
- 可选：Transformer 对比、参数敏感性或训练耗时图。

## 10. 最终交付与验收清单

### 必须完成（P0）

- [ ] 可重复读取 BCI IV 2a 的 9 名受试者四分类数据。
- [ ] 固定并记录滤波频段、时间窗、采样率、标签映射与 Session 划分。
- [ ] 预处理检查通过，数据统计和至少 3 张真实可视化图完整。
- [ ] CSP + LDA 完成 9 人实验并保存真实预测与指标。
- [ ] CSP + SVM 完成 9 人实验并保存真实预测与指标。
- [ ] EEGNet 完成 9 人实验，留有训练/验证记录与测试指标。
- [ ] 汇总 Accuracy、Macro-F1、9 人逐人结果及模型对比图。
- [ ] 有混淆矩阵、错误分析、模型局限性与结论。
- [ ] `README.md`、依赖版本、配置、脚本、日志及关键 CSV 可支持复现。
- [ ] 完成 Final Report PDF，所有图表与表格能追溯到实测结果。

### 争取完成（P1）

- [ ] 轻量 Transformer / EEG-Deformer 风格模型至少有一组可信的对比实验。
- [ ] 完成一组参数敏感性/对照实验（如 CSP 特征维数 4/8/12）。
- [ ] 增加训练耗时、参数量等复杂度对比。

## 11. 参考资料与范围说明

**课程资料**

- 老师提供的《数据挖掘 2026》：Project 3 — AI4Science：数字孪生脑与脑机接口（BCI）参考论文清单。
- 小组已提交的 Proposal（本规划沿用其明确的四分类任务和模型层级）。
- 小组中期汇报路线整理文本：《数据挖掘 Project 3：项目实施与中期 PPT 汇报大纲》。
- 《BCI 项目实施大纲》（13 页）：本规划的阶段验收、技术细节与 Final Report 结构依据。

**论文方向（老师提供的 Project 3 参考文献中的相关部分）**

- *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI*（LaBraM，ICLR 2024）
- *EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*（NeurIPS 2024）
- *EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces*（IEEE JBHI，2024/2025）
- *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*（ICLR 2025）

以上研究主要用于背景、Related Work、进阶模型设计与未来方向讨论。**本项目的最低交付是 EEG 四分类实验，而非复现完整脑基础模型、神经语音假体或数字孪生脑系统。**

---

**执行口径总结**：`统一数据处理 → CSP + LDA → CSP + SVM → EEGNet →（可选）轻量 Transformer → 模型对比与分析 → Final Report`。所有模型在相同跨 Session 四分类协议下评价；先保证 9 人核心结果真实、完整、可复现，再追求进阶方法。
