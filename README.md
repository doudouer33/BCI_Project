# BCI Competition IV 2a 运动想象四分类

阶段 A 提供 BNCI2014_001 的读取、预处理、固定划分和数据可视化。阶段 B 用同一协议实测 CSP + LDA 和 CSP + SVM 的 9 人基线。实验路线见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，统一参数见 [configs/experiment.yaml](configs/experiment.yaml)。

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

`requirements.txt` 覆盖数据流程和 CSP 基线。按规划一并准备的深度学习环境已在 Linux x86_64 CPU 上验证导入，版本为 PyTorch `2.11.0+cpu`、TorchAudio `2.11.0+cpu` 和 Braindecode `2.0.0`。需要复现这部分环境时可执行：

```bash
python -m pip install -c requirements.txt \
  --extra-index-url https://download-r2.pytorch.org/whl/cpu \
  'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu' 'braindecode==2.0.0'
```

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
