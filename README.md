# BCI Competition IV 2a 运动想象四分类

阶段 A 提供 BNCI2014_001 的读取、预处理、固定划分和数据可视化。实验路线见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，统一参数见 [configs/experiment.yaml](configs/experiment.yaml)。

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

`requirements.txt` 覆盖数据流程。按规划一并准备的深度学习环境已在 Linux x86_64 CPU 上验证导入，版本为 PyTorch `2.11.0+cpu`、TorchAudio `2.11.0+cpu` 和 Braindecode `2.0.0`。需要复现这部分环境时可执行：

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

**数据泄漏检查。** 连续信号按 Run 独立滤波，程序验证每个截取区间两侧都有 FIR 所需的采样点，且相邻试次使用的原始采样点互不重叠。阶段 A 只执行固定变换。后续标准化器或特征提取器须在 `train_idx` 对应的训练集上拟合。波形和功率谱仅使用训练集试次，测试 Session 参与完整性检查和类别计数。

## 共享 Python 接口

```python
from src.load_data import get_split_indices, load_config, load_subject

config = load_config()
X, y, metadata = load_subject(1, config)
train_idx, val_idx, test_idx = get_split_indices(metadata, config)
X_train, y_train = X[train_idx], y[train_idx]
```

`X` 维度为 `(trial, channel, time)`，`y` 为整数类别。`metadata` 与 `X` 的试次顺序一致，包含受试者、Session、Run、稳定的 `trial_id`、Run 内从零计数的 `event_sample` 和 `cue_sample`、事件码、类别名及伪迹标记。`metadata.attrs` 提供通道顺序、采样率、时间窗和逐 Run 检查。划分索引是每名受试者在此顺序中的零基位置，也随 `trial_id` 保存到 CSV。

## 输出与实测结果

- [数据统计](results/data_statistics.json) 保存有效配置、运行版本、逐受试者的类别与划分数量、逐 Run 的 EEG 完整性、cue 偏移和伪迹数量。
- [试次划分](results/trial_splits.csv) 保存每条试次的索引、标签、Session、Run、事件位置及所属集合。
- [EEG 时域图](results/figures/eeg_waveform.png) 展示 Subject 1 最早进入训练集的试次在 C3、Cz 和 C4 上的波形，幅值单位为 µV。
- [四类样本分布](results/figures/class_distribution.png) 展示全部已处理受试者按训练、验证、测试堆叠的类别计数。
- [EEG 功率谱](results/figures/eeg_psd.png) 使用 Subject 1 训练集的 C3、Cz、C4 估计各类别的平均 Welch 功率谱，Hann 窗长 2 秒、重叠 50%。

Subject 1 实测得到 `X.shape == (576, 22, 1000)`。原始 EEG Run 和预处理数组均为 `NaN=0`、`Inf=0`。每场 Session 有 Run `0`–`5`，每 Run 48 个试次，每场每类 72 个试次。训练、验证、测试数量分别为 230、58、288。

Subject 1 的通道顺序为 `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`。`EOG1`–`EOG3` 和 `STI` 在 MOABB Raw 对象中可检查，模型输入仅选取 EEG。

9 名受试者实测合计 108 个 Run、5,184 个试次，四类各 1,296 个。各人数组均为 `(576, 22, 1000)`。全部原始 EEG Run 和输出数组中的 NaN、Inf 均为 0。训练、验证、测试总数依次为 2,070、522、2,592。共有 488 个试次带源数据伪迹标记，均保留并写入 CSV。

在另一套仅安装 `requirements.txt` 的 Python 3.12.3 环境中重新运行，统计 JSON、划分 CSV 和三张 PNG 与上述输出逐字节一致。
