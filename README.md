# PlantVillage MSF-Res2Net SNN

脉冲植物病害分类项目：**首次时间编码（First Spike Coding）** + **MSF 多阈值神经元** + **Res2Net 多尺度结构**（通道维 split/concat，`dim=2`），基于 PyTorch 2.x 与 [SpikingJelly](https://github.com/fangwei123456/spikingjelly) 的 `activation_based` 多步模式（张量形状 `[T, B, C, H, W]`）。

## 环境

```bash
cd plantvillage_snn
pip install -r requirements.txt
```

依赖：`torch`、`torchvision`、`spikingjelly`、`datasets`（Hugging Face）、`tensorboard`、`scikit-learn` 等。

## 数据集准备

推荐使用 Hugging Face（与 [PlantVillage 官方仓库说明](https://github.com/spMohanty/PlantVillage-Dataset) 一致），首次运行会自动下载：

- 数据集卡片：[mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
- 代码中默认：`load_dataset("mohanty/PlantVillage", "color")`，带 **80/20 train/test** 且按叶片分组防泄漏。

若需本地克隆原始仓库（可选）：

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
```

训练脚本仍通过 `datasets` 在线加载；类数会在启动时从 HF 元数据自动推断并覆盖 `config.TrainConfig.num_classes`。

## 首次时间编码与输入归一化

数据管线使用 ImageNet 风格的 `Normalize`。编码前在 `utils/train_utils.denormalize_to_01` 中**近似反归一化并截断到 [0,1]**，再调用 `first_spike_coding` 得到 `[T,B,3,H,W]` 的输入脉冲。

## 训练

```bash
cd plantvillage_snn
python train.py --run_name exp_msf --epochs 100 --batch_size 32 --lr 0.1
```

常用参数：

- `--preset no_msf`：消融，使用标准 `LIFNode` 替代 MSF（见 `config.PRESETS`）。
- `--no_amp`：关闭混合精度。
- `--no_spike_counter`：关闭脉冲统计钩子。
- `--viz`：按 `config.TrainConfig.viz_every_n_epochs` 保存输入脉冲栅格与首个 MSF 单元的膜电位曲线（`utils/visualizer.py`）。

TensorBoard：

```bash
tensorboard --logdir runs/plantvillage_snn
```

检查点默认保存为 `checkpoints/best_msf_res2net.pt`。

## 测试与报告

```bash
python test.py --ckpt checkpoints/best_msf_res2net.pt --batch_size 64
```

输出整体准确率、全局平均脉冲率（`SpikeCounter`）以及 `sklearn` 的 precision / recall / F1 分类报告。

## 消融开关（`config.py`）

- `use_msf: bool`：`False` 时所有 `SeqConvBnMSF` 与块内节点退化为 **LIF**。
- `use_attention: bool`：`False` 时残差块内 **CSA 注意力关闭**（`nn.Identity`）。

可在 `config.PRESETS` 中组合 `no_msf`、`no_attention`、`ablation_baseline` 等，或通过 `get_config("no_msf")` 在代码里切换。

## 能耗相关指标

`utils/train_utils.SpikeCounter` 在 `MSFNode` 与 `LIFNode` 上注册 hook，按 epoch 统计：

- 全局平均脉冲率：`total_spikes / total_elements`（元素对应 `[T,B,C,H,W]` 输出位点）。
- 各层平均脉冲率：`per_layer_spike_rate`（便于论文中分层对比）。

训练日志与 TensorBoard 同时记录 train/val 的全局脉冲率。

## 项目结构

```
plantvillage_snn/
├── models/
│   ├── __init__.py
│   ├── msf_neuron.py
│   ├── attention.py
│   └── res2net_msf.py
├── data/
│   └── dataloader.py
├── utils/
│   ├── __init__.py
│   ├── train_utils.py    # first_spike_coding, SpikeCounter, cosine_lr, ...
│   └── visualizer.py
├── train.py
├── test.py
├── config.py
├── requirements.txt
└── README.md
```

## 参考

- PlantVillage 与 HF 用法：[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)、[Hugging Face PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
- MSF 多突触脉冲神经元相关：Fan, L., et al. *Nature Communications* (2025)（请按正式出版信息引用）

## GPU

安装对应 CUDA 的 PyTorch 后即可在 `config.device = "cuda"` 下训练；脚本在无 GPU 时会退回 CPU（速度较慢）。

## Kaggle Notebook

仓库内提供 **`notebooks/kaggle_plantvillage_snn.ipynb`**：

- 在 Kaggle 中 **Add Data** 添加你的 PlantVillage 数据集（本地文件夹结构：类名为子目录；或含 `train/`、`val/` 的划分）。
- 在 Notebook 里修改 **GitHub 克隆地址** 与 **`INPUT_NAME`**（与 `/kaggle/input/<名称>` 一致）。
- 数据管线使用 **`data/kaggle_dataloader.py`**：可在数据集根目录下自动查找合适的 ImageFolder 根（优先 `color` 等）。

也可不用 git，将本仓库打成 zip 用 *Upload* 作为 Input，再在 Notebook 中 `os.chdir` 到解压后的 `plantvillage_snn` 路径。
