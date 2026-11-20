# BCI Competition IV 2a 运动想象四分类

阶段 A 提供 BNCI2014_001 的读取、预处理、固定划分和数据可视化。阶段 B 用同一协议实测 CSP + LDA 和 CSP + SVM 的 9 人基线。阶段 C 加入 EEGNet。阶段 D 加入 Light Transformer，在全部 9 名受试者上比较四种模型的跨 Session 测试结果。实验路线见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，统一参数见 [configs/experiment.yaml](configs/experiment.yaml)。

## 环境与运行

在仓库根目录使用 Python 3.12。`requirements.txt` 固定了阶段 A 实测环境的依赖版本。系统 Python 缺少 `ensurepip` 时，可用 `uv venv --python 3.12 .venv` 创建环境，激活后执行 `uv pip install -r requirements.txt`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Subject 1 单独验收
python -m src.load_data --subjects 1 --output-dir results/subject_1

# 处理配置中的全部 9 名受试者
python -m src.load_data
```

首次运行时，MOABB 从数据发布方下载 `.mat` 文件至 `data/raw/`，全量缓存约需 750 MiB。可用 `--data-dir /path/to/cache` 指定缓存位置，可用 `--output-dir /path/to/results` 指定输出位置。`requirements.txt` 中锁定的 MNE 1.13.2 会为 MOABB 使用的 `standard_1005` 电极模板发出弃用提示，实测读取和波形处理均可完成。

`requirements.txt` 覆盖数据流程和 CSP 基线。EEGNet 的额外依赖和安装命令见[阶段 C](#阶段-ceegnet)。

## 统一数据协议

| 项目 | 设置 |
| --- | --- |
| 任务 | 每名受试者独立做跨 Session 四分类 |
| 输入 | 22 个 EEG 通道，原始采样率 250 Hz，数组单位 V |
| 截取 | 相对 cue `[0, 4)` 秒。首末采样点为 0.000 秒和 3.996 秒 |
| 滤波 | MNE 按 Run 对连续 EEG 做 8–30 Hz 零相位 Hamming FIR，`firwin`，低高端过渡带分别为 2、7.5 Hz，413 taps |
| 划分 | `0train` 内按类别分层划分训练集和 20% 验证集，种子 `42`。`1test` 整场用于最终测试 |
| 伪迹 | 保留 MOABB 读取的所有试次，在 `artifact_flag` 中记录 BNCI 伪迹标记 |

| 类别 | MOABB 事件码 | 项目标签 `y` |
| --- | ---: | ---: |
| 左手 `left_hand` | 1 | 0 |
| 右手 `right_hand` | 2 | 1 |
| 双脚 `feet` | 3 | 2 |
| 舌头 `tongue` | 4 | 3 |

**cue 核对。** 官方 GDF 将试次开始标为 `768`，训练场 cue 标为 `769`–`772`，测试场 cue 标为 `783`。实测 Subject 1 的两场 GDF 中共 576 个 cue，都比试次开始晚 500 个采样点。BNCI MAT 中的试次位置与对应 GDF 的逐试次间距一致，训练场类别也一致。MOABB 1.7.2 的 `STI` 是试次开始触发，类名 annotation 已向后移 500 点。共享加载器逐 Run 核对两组事件的位置和标签，直接按 annotation 切窗。 [官方数据说明](https://www.bbci.de/competition/iv/desc_2a.pdf) · [官方 GDF 下载](https://bbci.de/competition/iv/download/) · [MOABB 数据集文档](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)

**MOABB API 差异。** 在 Subject 1 上实测 `MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0, tmax=4)` 的 `Epochs` 时间轴为相对试次开始的 `[2, 6]` 秒，两端计入，共 1001 点，数组接口的单位为 µV。该版本的 `MotorImagery` 使用 IIR。本项目通过 `BNCI2014_001.get_data()` 读取连续 EEG 和事件，再用上表的 FIR 滤波和 `[0, 4)` 秒窗口分段，共 1000 点，保留 MNE 信号的 V 单位。Session 名称实测为 `0train`、`1test`。 [MOABB MotorImagery 文档](https://moabb.neurotechx.com/docs/generated/moabb.paradigms.MotorImagery.html)

**数据泄漏检查。** 连续信号按 Run 独立滤波，程序验证每个截取区间两侧都有 FIR 所需的采样点，且相邻试次使用的原始采样点互不重叠。阶段 A 只执行固定变换。波形和功率谱仅使用训练集试次，测试 Session 参与完整性检查和类别计数。

## 共享 Python 接口

```python
from src.load_data import get_split_indices, load_config, load_subject

config = load_config()
X, y, metadata = load_subject(1, config)
train_idx, val_idx, test_idx = get_split_indices(metadata, config)
X_train, y_train = X[train_idx], y[train_idx]
```

`X` 维度为 `(trial, channel, time)`，`y` 为整数类别。`metadata` 与 `X` 的试次顺序一致，包含受试者、Session、Run、稳定的 `trial_id`、Run 内从零计数的 `event_sample` 和 `cue_sample`、事件码、类别名及伪迹标记。`metadata.attrs` 提供通道顺序、采样率、时间窗和逐 Run 检查。划分索引是每名受试者在此顺序中的零基位置，也随 `trial_id` 保存到 CSV。

## 阶段 A 输出与实测结果

- [数据统计](results/data_statistics.json) 保存有效配置、运行版本、逐受试者的类别与划分数量、逐 Run 的 EEG 完整性、cue 偏移和伪迹数量。
- [试次划分](results/trial_splits.csv) 保存每条试次的索引、标签、Session、Run、事件位置及所属集合。
- [EEG 时域图](results/figures/eeg_waveform.png) 展示 Subject 1 最早进入训练集的试次在 C3、Cz 和 C4 上的波形，幅值单位为 µV。
- [四类样本分布](results/figures/class_distribution.png) 展示全部已处理受试者按训练、验证、测试堆叠的类别计数。
- [EEG 功率谱](results/figures/eeg_psd.png) 使用 Subject 1 训练集的 C3、Cz、C4 估计各类别的平均 Welch 功率谱，Hann 窗长 2 秒、重叠 50%。

Subject 1 实测得到 `X.shape == (576, 22, 1000)`。原始 EEG Run 和预处理数组均为 `NaN=0`、`Inf=0`。每场 Session 有 Run `0`–`5`，每 Run 48 个试次，每场每类 72 个试次。训练、验证、测试数量分别为 230、58、288。

Subject 1 的通道顺序为 `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`。`EOG1`–`EOG3` 和 `STI` 在 MOABB Raw 对象中可检查，模型输入仅选取 EEG。

9 名受试者实测合计 108 个 Run、5,184 个试次，四类各 1,296 个。各人数组均为 `(576, 22, 1000)`。全部原始 EEG Run 和输出数组中的 NaN、Inf 均为 0。训练、验证、测试总数依次为 2,070、522、2,592。共有 488 个试次带源数据伪迹标记，均保留并写入 CSV。

在另一套仅安装 `requirements.txt` 的 Python 3.12.3 环境中重新运行，统计 JSON、划分 CSV 和三张 PNG 与上述输出逐字节一致。

## 阶段 B：CSP + LDA / CSP + SVM

两条 scikit-learn Pipeline 均使用 MNE 的多分类 CSP。CSP 根据四类训练试次的协方差学习空间滤波器，并用各分量的对数平均功率表示每个试次。LDA 直接使用这些特征做分类。SVM 先用 `StandardScaler` 缩放 CSP 特征，再使用 `SVC` 分类。

每名受试者使用阶段 A 相同的 230 个训练试次、58 个验证试次和 288 个测试试次。LDA 比较 CSP 分量数 `8、4、12`。SVM 使用相同的分量数候选，并比较 `linear`、`rbf` 两种核以及 `C=1、10`，共 12 个组合。RBF 使用 `gamma="scale"`。这些候选值记录在 [实验配置](configs/experiment.yaml) 中。

调参时，每个候选 Pipeline 只在 230 个训练试次上拟合，在 58 个验证试次上预测。每名受试者、每种模型选取验证集 Macro-F1 最高的参数，同分时沿用配置中的候选顺序。随后用完整的 `0train`（train + val，共 288 个试次）重新拟合整条 Pipeline，再于 `1test` 上进行一次最终评估。CSP 和 SVM 的缩放器均随 Pipeline 拟合，测试阶段只调用 `predict`。

在仓库根目录、已安装 `requirements.txt` 的环境中运行。

```bash
# 验证 Subject 1
python -m src.train_csp --config configs/experiment.yaml --subjects 1 --output-dir results/subject_1

# 运行配置中的全部 9 名受试者
python -m src.train_csp --config configs/experiment.yaml --output-dir results
```

`--data-dir` 可指定数据缓存，`--output-dir` 可为每次运行指定独立的结果目录。

### 阶段 B 输出

- 18 组逐人指标及所选参数保存在 [metrics.csv](results/metrics.csv)。9 人汇总保存在 [csp_summary.csv](results/csp_summary.csv)。
- 每个模型、每名受试者各有一个预测文件，包含 `trial_id`、`y_true` 和 `y_pred`。标签编码见统一数据协议。[逐试次预测](results/predictions/)
- 运行命令、有效配置、依赖版本、各候选的验证 Macro-F1、所选参数及 18 组四分类混淆矩阵均可查阅。[运行日志](results/logs/csp_run.json)
- 两张图分别展示逐人测试准确率和 Subject 1 的四分类混淆矩阵。矩阵行表示真实类别，列表示预测类别，数值为试次数。[Accuracy 对比](results/figures/csp_accuracy.png) · [Subject 1 混淆矩阵](results/figures/csp_confusion_s1.png)

### 阶段 B 实测结果

| Subject | CSP + LDA Accuracy | CSP + LDA Macro-F1 | CSP + SVM Accuracy | CSP + SVM Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.40% | 67.75% | 65.97% | 63.69% |
| 2 | 53.47% | 48.58% | 51.04% | 45.91% |
| 3 | 70.14% | 67.64% | 63.89% | 63.42% |
| 4 | 58.68% | 53.85% | 48.61% | 46.24% |
| 5 | 37.15% | 32.17% | 31.60% | 23.89% |
| 6 | 46.18% | 44.05% | 42.71% | 39.62% |
| 7 | 59.72% | 52.67% | 48.61% | 38.19% |
| 8 | 76.74% | 76.30% | 71.88% | 71.48% |
| 9 | 74.65% | 72.59% | 63.89% | 61.15% |

下表按受试者等权计算，均值以百分比表示，标准差以百分点表示（总体标准差，`ddof=0`）。Macro-F1 为四个类别 F1 的算术平均。

| 模型 | Accuracy（均值 ± 标准差） | Macro-F1（均值 ± 标准差） |
| --- | ---: | ---: |
| CSP + LDA | 60.57 ± 12.57 | 57.29 ± 13.85 |
| CSP + SVM | 54.24 ± 12.27 | 50.40 ± 14.57 |

在本次固定协议下，LDA 在全部 9 名受试者上的 Accuracy 和 Macro-F1 都高于 SVM，两项均值分别高出 6.33 和 6.89 个百分点。两模型在 Subject 8 上准确率最高，在 Subject 5 上最低。Subject 1 的矩阵显示，LDA 和 SVM 分别有 31 和 29 个右手试次被判为左手，分别有 25 和 44 个双脚试次被判为舌头。

## 阶段 C：EEGNet

EEGNet 是轻量的 EEG 卷积网络。时间卷积先提取波形特征，逐深度空间卷积学习 22 个 EEG 通道的组合，可分离卷积进一步处理特征。实现使用 `braindecode.models.EEGNet` 2.0.0，启用该版本推荐的带最大范数约束的分类层，其余网络参数使用默认值。模型接收 `(batch, 22, 1000)` 的 `float32` 张量，输出 `(batch, 4)` logits。标签为 `int64`，交叉熵直接接收 logits。训练、验证和测试统一将阶段 A 的 V 值乘 `1e6` 转成 µV，无需拟合标准化统计量。 [Braindecode EEGNet 文档](https://braindecode.org/stable/generated/braindecode.models.EEGNet.html)

每名受试者用种子 `42` 独立初始化模型，以 DataLoader 打乱 230 个训练试次，使用 Adam 更新参数。Batch size 为 `32`，学习率为 `0.001`，最多训练 `100` 轮。每轮记录训练和验证的交叉熵与 Accuracy，以 58 个验证试次上的最低交叉熵保存最佳 checkpoint，同分时保留较早轮次。验证 Loss 连续 `15` 轮未下降则早停。最终恢复最佳 checkpoint，再构建 `1test` 的 DataLoader 并评估全场 288 个试次。作为对照，阶段 B 选定超参数后使用完整 `0train` 重拟合，EEGNet 保留由验证集选定的模型，其梯度更新只使用 230 个训练试次。

### 复现命令

沿用上文的 Python 3.12 环境和 `requirements.txt`。使用 GPU 时安装 CUDA 13.0 构建，然后安装 Braindecode。TorchAudio 2.11 支持 PyTorch 2.11 及后续版本。 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v2130) · [TorchAudio 兼容性说明](https://docs.pytorch.org/audio/main/installation.html#compatibility-matrix)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0+cu130' 'torchaudio==2.11.0+cu130'
python -m pip install -c requirements.txt 'braindecode==2.0.0'
```

纯 CPU 环境使用 CPU 构建。

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu'
python -m pip install -c requirements.txt 'braindecode==2.0.0'
```

```bash
# Subject 1，默认写入 results/subject_1
python -m src.train_eegnet --subjects 1

# 全部 9 名受试者，默认写入 results/
python -m src.train_eegnet
```

脚本默认在 PyTorch 可用时选用 CUDA，否则使用 CPU。可以用 `--device cpu` 或 `--device cuda` 显式选择。`--config`、`--data-dir` 和 `--output-dir` 可指定配置、数据缓存和输出路径。脚本固定训练与批次洗牌的种子，并启用 PyTorch 的确定性算法。

实测环境为 Linux x86_64、Python `3.12.3`。Subject 1 使用 `--subjects 1 --device cpu` 在 CPU 版 PyTorch 和 TorchAudio `2.11.0+cpu` 上先通过完整流程。正式九人实验使用 `--device cuda`，环境为 RTX 4090、驱动 `580.159.03`、CUDA `13.0`、cuDNN `9.20.0`、PyTorch `2.13.0+cu130`、TorchAudio `2.11.0+cu130` 和 Braindecode `2.0.0`。MOABB 1.7.2 加载电极模板时会出现阶段 A 已记录的 `standard_1005` 弃用提示。

### 阶段 C 输出

- [EEGNet 逐人指标](results/eegnet_metrics.csv)、[三模型逐人指标](results/model_metrics.csv)和[三模型汇总](results/model_summary.csv)。汇总按受试者等权计算，用总体标准差 `ddof=0`。
- `EEGNet_s1.csv` 至 `EEGNet_s9.csv` 保存 2,592 个测试试次的 `trial_id`、`y_true`、`y_pred`。[预测目录](results/predictions/)
- `EEGNet_s*_epochs.csv` 保存每轮训练和验证指标。`eegnet_run.json` 保存运行环境、配置、首批张量与梯度检查、最佳 Epoch、早停状态和逐人混淆矩阵。[日志目录](results/logs/) · [运行日志](results/logs/eegnet_run.json)
- `results/models/EEGNet_s*.pt` 保存最佳模型的 `state_dict`，由现有 `.gitignore` 排除。
- [Subject 1 训练曲线](results/figures/eegnet_training_s1.png) · [Subject 1 测试混淆矩阵](results/figures/eegnet_confusion_s1.png) · [三模型 Accuracy 对比](results/figures/model_accuracy.png) · [三模型 Macro-F1 对比](results/figures/model_macro_f1.png)

### 阶段 C 实测结果

| Subject | Accuracy | Macro-F1 | 最佳轮次 / 训练轮次 |
| ---: | ---: | ---: | ---: |
| 1 | 40.97% | 34.65% | 13 / 28 |
| 2 | 26.04% | 17.80% | 12 / 27 |
| 3 | 87.85% | 87.64% | 94 / 100 |
| 4 | 38.54% | 38.03% | 27 / 42 |
| 5 | 26.74% | 20.11% | 14 / 29 |
| 6 | 38.54% | 36.89% | 52 / 67 |
| 7 | 36.81% | 35.15% | 15 / 30 |
| 8 | 28.82% | 22.76% | 14 / 29 |
| 9 | 78.82% | 78.79% | 99 / 100 |

均值以百分比表示，标准差以百分点表示。

| 模型 | Accuracy（均值 ± 标准差） | Macro-F1（均值 ± 标准差） |
| --- | ---: | ---: |
| CSP + LDA | 60.57 ± 12.57 | 57.29 ± 13.85 |
| CSP + SVM | 54.24 ± 12.27 | 50.40 ± 14.57 |
| EEGNet | 44.79 ± 21.35 | 41.31 ± 23.59 |

EEGNet 在 Subject 3 和 9 上的 Accuracy 与 Macro-F1 均超过两条 CSP 基线，在其余 7 人上均低于两条基线。其平均 Accuracy 分别比 CSP + LDA 和 CSP + SVM 低 15.78 和 9.45 个百分点，受试者之间的波动也更大。

7 名受试者触发早停，Subject 3 和 9 达到 100 轮。Subject 1 从最佳轮次到停止时，训练 Loss 由 0.876 降至 0.618，验证 Loss 由 1.219 升至 1.336，呈现过拟合迹象。测试时有 63 个左手试次被判为右手，右手共占 183 个预测（63.54%），反映出该受试者的类别预测偏向。每人有限的训练样本与跨 Session 信号变化都可能影响泛化。

## 阶段 C 补充实验：EEGNet 标准化与 Refit

原始 EEGNet 在部分受试者上出现训练 Loss 下降、验证 Loss 回升的现象。本实验检查两项改动能否改善跨 Session 表现，即消除不同通道的总体幅值差异，以及在验证集选定训练轮次后，让全部训练 Session 试次参与参数更新。

| 实验 | 模型输入 | 最终训练方式 |
| --- | --- | --- |
| Baseline | µV | 复用原始 230 试次训练的最佳验证模型及其已保存结果 |
| Z-score | 由 230 个训练试次拟合的逐通道 Z-score | 230 个试次训练，恢复验证 Loss 最低的 checkpoint |
| Refit | µV | 全部 288 个 `0train` 试次，训练到原始日志中的最佳轮次 |
| Z-score + Refit | 由全部 288 个 `0train` 试次拟合的逐通道 Z-score | 全部 288 个试次，训练到 Z-score 组在 230/58 划分上选定的最佳轮次 |

共享数据加载器和试次清单锁定原有 `230 / 58 / 288` 划分，模型结构和训练配置沿用阶段 C。Z-score 先将 V 转成 µV，对每个通道沿试次和时间轴以 `float64` 计算总体均值和标准差，再应用 `(X − mean) / (std + 1e-6)`。模型输入为无量纲 `float32`，加载器会检查 NaN 和 Inf。Z-score 选轮次时，训练、验证及该组测试均使用 230 个训练试次的统计量。组合组的重新训练和最终测试均使用 288 个 `0train` 试次重新拟合的统计量。

两种 Refit 都从种子 `42` 重新初始化 EEGNet、Adam 和训练批次顺序，每轮遍历完整 `0train`，按选定的轮数结束。选轮次仍使用原有最低验证 Loss 规则和 15 轮耐心值。三个新增组均完成训练后才评估该受试者的 `1test`。历史实验已经评估过同一个测试 Session，以下比较属于探索性结果。配置在运行前固定。

使用阶段 C 的环境，在仓库根目录运行。

```bash
# Subject 1 验证，输出到 results/eegnet_ablation/subject_1/
.venv/bin/python -m src.train_eegnet_ablation --subjects 1 --device cuda

# 正式九人对照，输出到 results/eegnet_ablation/
.venv/bin/python -m src.train_eegnet_ablation --device cuda
```

新增脚本和独立配置复用原始训练循环及共享实验协议。正式运行沿用阶段 C 的 RTX 4090、PyTorch `2.13.0+cu130` 和 Braindecode `2.0.0`。 [训练脚本](src/train_eegnet_ablation.py) · [补充实验配置](configs/eegnet_ablation.yaml)

四组逐人指标包含最佳轮次、选轮阶段实际轮数和 Refit 实际轮数。运行日志保存有效配置、种子、逐通道统计量和首批更新检查。Z-score 组保存每人的训练及验证曲线和逐轮记录。两种 Refit 保存逐轮训练日志和逐试次预测。 [逐人指标](results/eegnet_ablation/metrics.csv) · [四组汇总](results/eegnet_ablation/summary.csv) · [预测目录](results/eegnet_ablation/predictions/) · [运行日志](results/eegnet_ablation/logs/run.json) · [Subject 1 的 Z-score 曲线](results/eegnet_ablation/figures/eegnet_training_s1.png) · [平均 Accuracy 对比图](results/eegnet_ablation/figures/mean_accuracy.png)

### 九人对照结果

每个单元格依次为测试 Accuracy 和 Macro-F1，均以百分比表示。

| Subject | Baseline | Z-score | Refit | Z-score + Refit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 40.97% / 34.65% | 25.69% / 11.62% | 31.25% / 22.50% | 25.35% / 10.92% |
| 2 | 26.04% / 17.80% | 25.35% / 19.81% | 28.47% / 18.68% | 27.08% / 22.85% |
| 3 | 87.85% / 87.64% | 25.00% / 10.00% | 86.81% / 86.35% | 25.35% / 13.87% |
| 4 | 38.54% / 38.03% | 34.38% / 31.89% | 42.01% / 38.66% | 39.58% / 36.22% |
| 5 | 26.74% / 20.11% | 30.21% / 25.90% | 24.65% / 15.06% | 25.69% / 12.44% |
| 6 | 38.54% / 36.89% | 37.85% / 36.48% | 27.78% / 22.46% | 30.21% / 24.92% |
| 7 | 36.81% / 35.15% | 30.56% / 24.68% | 26.04% / 12.34% | 25.00% / 10.08% |
| 8 | 28.82% / 22.76% | 25.00% / 12.01% | 26.74% / 15.28% | 25.35% / 11.45% |
| 9 | 78.82% / 78.79% | 76.04% / 75.58% | 79.51% / 79.15% | 80.56% / 80.79% |

下表按受试者等权计算。均值以百分比表示，标准差及变化量以百分点表示，总体标准差使用 `ddof=0`。

| 实验 | Accuracy 均值 ± 标准差 | Macro-F1 均值 ± 标准差 | Accuracy 相对 Baseline 变化 | Macro-F1 相对 Baseline 变化 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 44.79 ± 21.35 | 41.31 ± 23.59 | 0.00 | 0.00 |
| Z-score | 34.45 ± 15.32 | 27.55 ± 19.09 | −10.34 | −13.76 |
| Refit | 41.47 ± 22.86 | 34.50 ± 26.83 | −3.32 | −6.82 |
| Z-score + Refit | 33.80 ± 17.11 | 24.84 ± 21.42 | −11.00 | −16.48 |

Refit 单独使用时，Subject 2、4 和 9 的两项指标有所提升，九人均值均下降。在 Z-score 基础上加入 Refit 后，平均 Accuracy 和 Macro-F1 又分别下降 0.66 和 2.71 个百分点。固定 Epoch 数时，每轮的梯度更新次数随训练集规模从 8 批增至 9 批，Refit 的差异包含样本数和更新次数的共同变化。

Z-score 组选轮阶段有 8 人触发早停。Subject 1 从最佳第 11 轮到停止的第 26 轮，训练 Loss 从 0.959 降至 0.646，验证 Loss 从 1.341 升至 1.434，仍有过拟合迹象。Subject 3 在 Z-score 组把 288 个测试试次全部预测成左手，Accuracy 为 25.00%，Macro-F1 为 10.00%。检查中未出现 NaN、Inf 或失败的参数更新。以上观察来自单一种子和同一测试 Session。

## 阶段 D：轻量 Transformer

Light Transformer 借鉴 EEG-Deformer 的卷积与 Transformer 组合思路。课程实现采用两层卷积、固定长度的时间 token 和原生 PyTorch Encoder。卷积先学习局部波形和电极组合，池化将注意力序列从 1,000 个采样点缩短到 40 个 token。卷积中的时间核和池化窗口均为 25 个采样点，对应 100 ms。 [模型定义](src/models/light_transformer.py) · [实验配置](configs/experiment.yaml)

| 步骤 | 设置 | 输出形状 |
| --- | --- | --- |
| 输入 | 22 通道，250 Hz，V 转 µV 后使用 `float32` | `(B, 22, 1000)` |
| 时间卷积 | 16 个 `1 × 25` 卷积核，同长填充，BatchNorm | `(B, 16, 22, 1000)` |
| 逐深度空间卷积 | `22 × 1` 卷积核，每个时间滤波器对应 2 个空间滤波器，BatchNorm 和 ELU | `(B, 32, 1, 1000)` |
| 时间池化和投影 | 平均池化步长 25，映射至 64 维，加可学习位置编码和 Dropout | `(B, 40, 64)` |
| Transformer Encoder | 2 层，4 个注意力头，前馈层维度 128，GELU，Pre-LayerNorm | `(B, 40, 64)` |
| 分类 | 沿 token 维平均池化，线性层输出四类 logits | `(B, 4)` |

模型有 **73,204** 个可训练参数，位置编码和 Encoder 使用 `0.2` 的 Dropout。每名受试者独立用种子 `42` 初始化模型和批次洗牌，复用阶段 A 的 `load_subject()`、`get_split_indices()`、8–30 Hz FIR 和 cue 后 `[0, 4)` 秒窗口。脚本逐人核对试次索引、ID 和标签。µV 单位转换沿用原始 EEGNet，BatchNorm 的统计量由训练批次更新。

共用阶段 C 的 DataLoader、PyTorch 训练循环和早停逻辑。用 `0train` 中的 230 个试次训练，Adam 学习率 `0.001`，batch size `32`，最多 `100` 轮，损失为交叉熵。按 58 个验证试次上的最低 Loss 选 checkpoint，同值保留较早轮次，连续 15 轮无改善则停止。恢复 checkpoint 后构建测试 DataLoader，评估 `1test` 的 288 个试次，Refit 关闭。 [训练入口](src/train_transformer.py) · [共享训练逻辑](src/torch_training.py) · [固定试次清单](results/trial_splits.csv)

### 复现命令

沿用仓库根目录的 Python 3.12 环境和 `requirements.txt`，再安装对应设备的 PyTorch。正式运行使用 Linux x86_64、Python `3.12.3`、RTX 4090、驱动 `580.159.03`、PyTorch `2.13.0+cu130`、CUDA `13.0` 和 cuDNN `9.20.0`。运行启用 PyTorch 确定性算法。MOABB 1.7.2 沿用阶段 A 记录的电极模板，加载时会出现 `standard_1005` 弃用提示。

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.13.0+cu130'

# Subject 1，默认输出到 results/transformer/subject_1/
.venv/bin/python -m src.train_transformer --subjects 1 --device cuda

# 全部 9 名受试者，默认输出到 results/transformer/
.venv/bin/python -m src.train_transformer --device cuda
```

可以用 `--device cpu` 在 CPU 上运行，省略设备时自动选择。用 `--config` 指定配置、`--data-dir` 指定数据缓存、`--output-dir` 指定输出目录。实际运行先把 Subject 1 的完整验收输出写入单独的临时目录，再用同一配置从头训练 9 名受试者。 [正式运行日志](results/transformer/logs/transformer_run.json)

### 阶段 D 输出

- [逐人指标](results/transformer/transformer_metrics.csv) · [模型汇总](results/transformer/transformer_summary.csv) · [四模型逐人指标](results/transformer/four_model_metrics.csv) · [四模型汇总](results/transformer/four_model_summary.csv) · [参数量与耗时](results/transformer/transformer_runtime.csv)
- `LightTransformer_s*.csv` 记录 2,592 个测试试次的 `trial_id`、`y_true` 和 `y_pred`。逐轮 CSV 保存训练和验证的 Loss、Accuracy，运行日志保存四类混淆矩阵、环境和首批梯度及更新检查。最佳 checkpoint 保存为 `results/transformer/models/LightTransformer_s*.pt`，由 `.gitignore` 排除。 [预测目录](results/transformer/predictions/) · [日志目录](results/transformer/logs/)
- [Subject 1 训练曲线](results/transformer/figures/transformer_training_s1.png) · [Subject 1 测试混淆矩阵](results/transformer/figures/transformer_confusion_s1.png) · [四模型 Accuracy 对比](results/transformer/figures/four_model_accuracy.png) · [四模型 Macro-F1 对比](results/transformer/figures/four_model_macro_f1.png)

### 阶段 D 实测结果

| Subject | 测试 Accuracy | 测试 Macro-F1 | 最佳轮次 / 训练轮次 | 训练阶段耗时（秒） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 60.07% | 57.81% | 12 / 27 | 3.41 |
| 2 | 40.28% | 40.56% | 22 / 37 | 4.33 |
| 3 | 72.92% | 70.24% | 26 / 41 | 4.68 |
| 4 | 42.01% | 34.58% | 12 / 27 | 3.01 |
| 5 | 36.81% | 31.77% | 23 / 38 | 4.20 |
| 6 | 36.81% | 37.40% | 12 / 27 | 3.01 |
| 7 | 47.92% | 45.57% | 19 / 34 | 3.82 |
| 8 | 64.93% | 64.79% | 17 / 32 | 3.57 |
| 9 | 60.42% | 58.33% | 12 / 27 | 2.99 |

四个模型使用相同的 9 名受试者和逐人测试试次。EEGNet 一行来自阶段 C 原始 Baseline。受试者等权，均值用百分比表示，标准差用百分点表示（`ddof=0`）。 [阶段 C 原始比较数据](results/model_metrics.csv)

| 模型 | Accuracy（均值 ± 标准差） | Macro-F1（均值 ± 标准差） |
| --- | ---: | ---: |
| CSP + LDA | 60.57 ± 12.57 | 57.29 ± 13.85 |
| CSP + SVM | 54.24 ± 12.27 | 50.40 ± 14.57 |
| 原始 EEGNet | 44.79 ± 21.35 | 41.31 ± 23.59 |
| Light Transformer | 51.35 ± 12.70 | 49.01 ± 13.28 |

Light Transformer 相较原始 EEGNet 的平均 Accuracy 和 Macro-F1 分别提高 **6.56** 和 **7.69** 个百分点，两项都提升的受试者有 5 名。它在 Subject 3 上超过两条 CSP 基线，但平均 Accuracy 分别落后 LDA 和 SVM 9.22、2.89 个百分点，Macro-F1 分别落后 8.28、1.39 个百分点。CSP 的空间方差特征在这组小样本跨 Session 比较中整体得分更高，其最终拟合使用完整 `0train` 的 288 个试次。两种深度模型各用 230 个训练试次更新权重。

Transformer 的 Accuracy 从 36.81% 到 72.92%，受试者间的标准差比原始 EEGNet 小。9 人均触发早停。Subject 1 从最佳第 12 轮到停止的第 27 轮，训练 Loss 由 0.751 降至 0.391，验证 Loss 由 0.868 升至 0.994，符合过拟合迹象。跨 Session 测试中，72 个双脚试次有 48 个被判为舌头。这些观察来自每人 230 个训练试次、一个固定种子和一个测试 Session，无法单独量化样本量和 Session 变化各自的影响。

### 参数量与耗时

| 模型 | 可训练参数量 | 平均训练轮次 | 平均训练阶段（秒/人） | 平均端到端（秒/人） |
| --- | ---: | ---: | ---: | ---: |
| 原始 EEGNet | 3,444 | 50.22 | — | 9.82 |
| Light Transformer | 73,204 | 32.22 | 3.67 | 9.08 |

两次正式实验均使用 RTX 4090。Transformer 参数量约为 EEGNet 的 21.3 倍，实测训练阶段合计 33.02 秒，端到端平均耗时少 0.74 秒，平均训练轮次也更少。训练阶段包含训练和验证循环及 checkpoint 读写。端到端从逐人加载数据计至测试、日志及示例绘图完成。原始 EEGNet 日志只提供端到端时间，因此耗时比较采用该口径。 [耗时对比](results/transformer/transformer_runtime.csv) · [原始 EEGNet 运行日志](results/logs/eegnet_run.json)

CUDA 首批检查验证了 `(32, 22, 1000) → (32, 4)`、有限的交叉熵和梯度，以及 Adam 参数更新。CPU 单批也通过前向、反向和参数更新检查。Subject 1 两次运行的逐轮指标、checkpoint 张量和预测完全一致。正式九人实验从全部 2,592 个预测重算 Accuracy、Macro-F1 和混淆矩阵，与保存的指标和日志一致。
