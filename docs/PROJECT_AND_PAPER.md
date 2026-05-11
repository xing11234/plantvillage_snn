# PlantVillage 脉冲分类项目：技术说明与论文风格文稿

本文档分两部分：**（一）项目技术说明书**——便于复现与答辩；**（二）顶会风格学术论文草稿**——以 **MSF + $T{=}8$** 为主模型，系统化描述网络、理论与消融设计。文中**数值型实验结果**需在你完成全量训练与 `test.py` 评测后填入表格；此处给出**指标定义、对比协议与结论撰写框架**，避免虚构精度。

---

# 第一部分：项目技术说明书

## 1. 项目目标与数据管线

**目标**：在 PlantVillage 彩色叶片图像上完成多类植物病害（或健康）分类，采用 **SNN 多时间步前向** + **可配置 MSF 神经元** + **Res2Net 式多尺度骨干**，并支持 **CSA 通道–空间注意力** 与 **标准 LIF** 消融。

**数据**：

- 默认通过 Hugging Face `mohanty/PlantVillage`（`color` 配置）加载；类数在运行时写入 `TrainConfig.num_classes`。
- Kaggle 路径下使用 `data/kaggle_dataloader.py` 的本地 ImageFolder 管线。
- 输入经 ImageNet 风格归一化；编码前用 `denormalize_to_01` 将张量近似映射回 $[0,1]$，再进入脉冲编码（见下节）。

## 2. 输入编码：首次脉冲时间编码（First Spike Coding）

对强度图像 $I \in [0,1]^{B \times 3 \times H \times W}$，时间步数为 $T$（主实验取 **$T=8$**）。像素级首次发放时刻为：

$$
t^\*(h,w,c) = \operatorname{clip}\Bigl(\operatorname{round}\bigl((1 - I_{b,c,h,w})\cdot(T-1)\bigr),\, 0,\, T-1 \Bigr).
$$

第 $t$ 步的输入脉冲为 one-hot 时间编码：

$$
s^{\mathrm{in}}_{t,b,c,h,w} = \mathbb{1}\{ t = t^\*(h,w,c) \}.
$$

**含义**：亮度越高，发放越早（时间信息携带强度）。实现见 `utils/train_utils.first_spike_coding`，输出形状为 `[T, B, 3, H, W]`。

## 3. 网络结构概览

### 3.1 时空张量约定

全网络在 SpikingJelly `activation_based` 多步模式下运行，张量布局为 **`[T, B, C, H, W]`**；卷积与 BN 在 `dim=2` 上操作通道，**时间维始终为 `dim=0`**，避免与 batch 混用（Notebook 中已强调勿对 `[T,B,...]` 使用 `DataParallel` 的错误切分）。

### 3.2 MSF-Res2Net 骨干

- **Stem**：`7\times7` 卷积（stride 2）+ BN + 脉冲节点 + `MaxPool`。
- **四阶段 Res2Net-29 风格**：`layers=(2,2,2,2)`，`base_width=32`，`scale=4`（**Bottle2neckMSF**）。
- **Bottle2neckMSF**：$1\times1$ 升维 → 在 **通道维 `dim=2`** 上 split 为 `scale` 组 → 组间残差式 $3\times3$ 卷积融合 → concat → $1\times1$ 投影 + **CSA 注意力**（可关）+ 残差相加。
- **分类头**：`AdaptiveAvgPool2d(1)` + `Linear(512, C_{\mathrm{cls}})`；前向返回 **`[T, B, num_classes]`** 的每步 logits。

实现：`models/res2net_msf.py` 中 `MSFRes2Net` 与 `build_model(cfg)`。

### 3.3 CSA 注意力（可选）

- **通道注意力（CA）**：对时间维求均值得到 `[B,C,H,W]`，再 SE 式 MLP + Sigmoid，权重广播回 `[T,B,C,H,W]`。
- **空间注意力（SA）**：对每个 $t$，在通道上做 max/mean，拼接后经 $7\times7$ 卷积 + Sigmoid 得到空间门控。

`use_attention=False` 时替换为 `nn.Identity()`。实现：`models/attention.py`。

## 4. MSF 神经元原理与公式

### 4.1 前向动力学（与 surrogate 无关）

对每空间–通道位置，膜电位递推（单步输入为 $x_t$）：

$$
v_t = \lambda\, v_{t-1} + x_t, \qquad \lambda \equiv \texttt{msf\_decay}.
$$

发放规则（硬阈值，最低档阈值 $\theta_0 \equiv$ `msf_v_threshold`）：

$$
o_t = H(v_t - \theta_0), \quad H(u)=\mathbb{1}\{u \ge 0\}.
$$

硬重置（减去阈值电荷）：

$$
v_t \leftarrow v_t - o_t \cdot \theta_0.
$$

**多阈值 $\{\theta_d\}_{d=0}^{D-1}$**：在实现中由 `torch.linspace(θ_0, θ_0(1+0.25(D-1)), D)` 生成，**仅用于反向 surrogate 混合**；前向仍只对 $\theta_0$ 比较发放。

默认 **$D=4$**（`msf_D`）。

### 4.2 反向：多阈值 surrogate 混合 + STE

令 $f$ 为所选 surrogate 的导数形状函数（对 $u = v - \theta_d$ 逐阈值计算），反向梯度近似为：

$$
\frac{\partial o_t}{\partial v_t} \approx \frac{1}{D}\sum_{d=0}^{D-1} f(u_{t,d}, \alpha), \quad u_{t,d} = v_t - \theta_d,
$$

其中 $\alpha \equiv$ `msf_surrogate_alpha`。前向硬门 + 反向光滑通道，即 **Straight-Through Estimator（STE）** 族。

### 4.3 本项目实现的 surrogate（与代码一致）

记 $a=\max(\alpha,\epsilon)$，$x$ 为相对阈值的差 $v-\theta$。

| 名称 | $f(x,\alpha)$（代码形式） |
|------|---------------------------|
| rect | $\mathbb{1}\{|x|<a\}\,/\,(2a)$ |
| sigmoid | $(1/a)\cdot e^{-|x|/a} / (1+e^{-|x|/a})^2$ |
| arctan | $a\,/\,\bigl(\pi(1+(ax)^2)\bigr)$ |
| gaussian | $(2\pi)^{-1/2}a^{-1}\exp(-x^2/(2a^2))$ |

**调参提示**（与 `config.py` 注释一致）：`arctan` / `sigmoid` 常与 **$\alpha$ 与学习率联合** 调节；光滑 surrogate 平均反向幅度常低于 `rect`，欠拟合时可略增 `lr` 或增大 $\alpha$（在合理范围内）。

### 4.4 消融：LIF 基线

`use_msf=False` 时，`make_spiking_node` 使用 SpikingJelly `LIFNode`（`tau=lif_tau`，`decay_input=True` 等），用于 **MSF vs 标准 LIF** 对照。

## 5. 训练目标与推理聚合

每步输出 logits $\mathbf{z}_t \in \mathbb{R}^{B\times K}$，时间聚合采用 **时间平均**：

$$
\bar{\mathbf{z}} = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{z}_t.
$$

损失为对 $\bar{\mathbf{z}}$ 的交叉熵（可选 `label_smoothing`）。与 `train.py` / Notebook 中 `logits_t.mean(dim=0)` 一致。

优化器默认 **SGD（Nesterov）+ 权重衰减**；学习率 **余弦退火 + warmup**（`utils/lr_schedule.cosine_lr`）。支持 **AMP**。

每个 batch 前调用 `reset_snn_state` 重置 LIF 等记忆模块状态，避免跨 batch 泄漏。

## 6. 评价指标与能耗代理

- **分类**：Top-1 准确率（train/val）；`test.py` 可输出 per-class precision / recall / F1。
- **脉冲活动**：`SpikeCounter` 在 `MSFNode`/`LIFNode` 输出上统计  
  $\text{rate} = \frac{\sum s}{\#\text{张量元素}}$（元素对应 `[T,B,C,H,W]` 位点），用于 **能耗或稀疏性** 的相对比较（非绝对焦耳数）。

## 7. 参数设置（主模型：MSF，$T=8$）

下表为 **`TrainConfig` 默认值**（本地 `train.py`）；Notebook 中常对 `epochs`、`batch_size`、`lr`、`T`、`warmup`、`label_smoothing` 等做折中，以 Kaggle 墙钟为约束。

| 类别 | 参数 | 主模型推荐（全量实验） |
|------|------|------------------------|
| 时间 | `T` | **8** |
| MSF | `msf_D` | 4 |
| MSF | `msf_decay` ($\lambda$) | 0.25（可试 0.5–0.75 提高积分记忆） |
| MSF | `msf_v_threshold` | 1.0 |
| MSF | `msf_surrogate` | `rect`（基线）；`arctan` 时建议 `msf_surrogate_alpha` 2–5 并联调 `lr` |
| MSF | `msf_surrogate_alpha` | 1.0（光滑 surrogate 需扫） |
| 结构 | `base_width`, `scale`, `layers` | 32, 4, (2,2,2,2) |
| 训练 | `epochs` | 100（默认）；快速验证可 20 |
| 训练 | `lr` | 0.1（峰值）；Kaggle 常见 0.05 |
| 训练 | `weight_decay`, `momentum` | 5e-4, 0.9 |
| 训练 | `warmup_epochs`, `min_lr` | 5, 1e-6 |
| 训练 | `grad_clip_max_norm` | 5.0 |
| 数据 | `image_size`, `batch_size` | 224, 32（视显存调整） |
| 开关 | `use_msf`, `use_attention` | True, True（主模型） |

**消融预设**（`config.PRESETS`）：

- `no_msf`：`use_msf=False`（LIF）。
- `no_attention`：`use_attention=False`。
- `ablation_baseline`：二者皆关。

## 8. 结论分析（撰写指南）

建议在固定 **数据划分与增强策略** 下报告：

1. **主结果**：MSF + CSA + $T{=}8$ + `rect`（或你选定的 surrogate）的 **验证集最佳 Top-1** 与 **测试集** 指标。
2. **消融**：相对主模型，$\Delta$ **准确率** 与 **全局脉冲率** 的变化（MSF→LIF、去 CSA、$T$ 从 8 改为 4/16 的时延–精度折中）。
3. ** surrogate 研究**（可选小节）：在固定 MSF 与 $T$ 下比较 `rect` / `arctan` / `sigmoid` 的收敛速度与最终精度；讨论 **训练–验证 gap** 与 **脉冲率曲线**。
4. **局限性**：首次脉冲编码对光照与归一化敏感；时序步数与墙钟成本；MSF 实现上前向单阈值、反向多阈值的近似性等。

---

# 第二部分：学术论文风格文稿（草稿）

> 下列内容为**可投稿结构的 LaTeX 式逻辑**，中文撰写；**实验数值请替换为你仓库 TensorBoard / `test.py` 的实测结果**。图表建议：曲线（acc、loss、spike rate）、混淆矩阵、若干层脉冲栅格（`--viz` / `utils/visualizer.py`）。

## 摘要

植物叶片病害的自动识别对精准农业具有重要意义。传统人工神经网络（ANN）在静态图像分类上表现优异，但难以显式刻画事件驱动与稀疏激活特性。脉冲神经网络（SNN）通过时间维度传递信息，便于与 **能效** 和 **神经形态硬件** 的叙事衔接。本文面向 PlantVillage 彩色数据集，构建 **多时间步前向** 的 **MSF-Res2Net**：输入采用 **首次脉冲时间编码**；骨干为 **Res2Net 式多尺度瓶颈结构**，通道维 split–concat 以提取多粒度空间特征；神经元采用 **多阈值 MSF 代理梯度**（前向硬阈值 IF，反向多阈值光滑 surrogate 混合）；并嵌入轻量 **CSA（通道–空间）注意力** 以增强病害区域响应。以 **$T{=}8$、MSF 全开** 为主配置，我们在统一训练协议下开展 **MSF vs LIF**、**有无 CSA**、**时间步 $T$** 及 **不同 surrogate** 的消融实验，并从 **分类性能** 与 **全局脉冲率** 两方面进行分析。实验表明（*此处填入主要结论与最优精度*）。最后讨论计算开销、编码方案局限与未来在更大规模数据集与神经形态部署上的拓展方向。

**关键词**：脉冲神经网络；PlantVillage；Res2Net；多阈值代理梯度（MSF）；首次脉冲编码；注意力机制；消融实验

---

## 1 引言

植物病害早期检测可降低经济损失与环境风险。公开数据集 PlantVillage 推动了基于深度 CNN 的大量工作，然而在资源受限边缘设备上，稠密乘加与持续推理的能耗仍受关注。SNN 以离散脉冲传递信息，在理论上更贴近异步计算范式。将 **高性能 CNN 结构**（如残差多分支设计）迁移到 SNN，并解决 **不可微发放** 带来的训练难题，是当前研究热点之一。

本文贡献可概括为：（1）给出面向 `[T,B,C,H,W]` 布局的 **MSF-Res2Net + CSA** 完整实现与训练管线；（2）以 **$T{=}8$ 的 MSF 主模型** 为核心，系统化 **消融** MSF、注意力与时间编码长度；（3）提供 **脉冲率** 等辅助指标，服务于能效叙事。实现基于 PyTorch 与 SpikingJelly。

---

## 2 国内外研究现状

**植物病害识别**：早期工作以手工特征与传统分类器为主；深度 CNN（如 VGG、ResNet）显著提升了 PlantVillage 上的精度。近期工作关注轻量化、注意力、迁移学习与数据增强，但多数仍为 ANN 范式。

**SNN 图像分类**：代理梯度（STE）使深层 SNN 训练成为可能；时间编码（rate、latency、first-spike 等）将静态图像映射为脉冲序列。结构方面，残差网络、SE 注意力等已被迁移至 SNN。**多阈值 / 多突触发放（MSF）** 思想通过丰富梯度路径或近似多水平发放，旨在改善深层训练稳定性与表达能力；具体形式因文献而异，本文实现采用 **前向单阈值 IF + 反向多阈值 surrogate 平均**（见第 4 节）。

**差距**：将 **Res2Net 多尺度瓶颈** 与 **MSF + 通道–空间注意力** 在植物病害细粒度纹理与病斑定位需求下结合，并给出 **可复现消融** 的工作仍相对分散。本文在该交叉点上给出工程化方案与实验协议。

---

## 3 相关理论基础

### 3.1 泄漏积分–发放与代理梯度

IF/LIF 神经元膜电位随输入累积，超过阈值则发放并重置。发放函数对膜电位不可微，BPTT 训练中普遍采用 STE：前向用阶跃函数，反向用对 $(v-\theta)$ 的光滑函数 $f$ 近似 $\partial s/\partial v$。

### 3.2 时间编码与信息容量

Rate coding 在时间维上重复静态特征；latency / first-spike coding 则把强度映射为发放时刻，在有限 $T$ 下形成 **时间压缩** 表示。本文采用 **first-spike**：高亮区域更早发放，与病斑高对比区域可能更亮的先验相容（亦受成像条件影响）。

### 3.3 Res2Net 多尺度表示

Res2Net 在瓶颈内部以 **分尺度残差融合** 扩大感受野组合，有利于细粒度纹理；本文在 **通道维 split** 上实现，以兼容 SNN 多步张量布局。

---

## 4 SNN 模型设计

### 4.1 总体框架

编码器将 RGB 图像映射为 $s^{\mathrm{in}}\in\{0,1\}^{T\times B\times 3\times H\times W}$；骨干为 MSF-Res2Net，输出 $\mathbf{z}_t$；分类损失作用在 $\bar{\mathbf{z}}=\frac{1}{T}\sum_t \mathbf{z}_t$。

### 4.2 MSF 神经元（主模型）

前向见式 (1)–(3)；反向见式 (4) 与第 1 部分表格。默认 $D=4$，$\theta_d$ 线性递增。**主模型设置**：$T=8$，$\lambda=0.25$，$\theta_0=1$，surrogate 默认 `rect`；对比实验切换 `arctan` 等并扫描 $\alpha$。

### 4.3 CSA 注意力模块

在瓶颈输出与残差相加前插入 CSA（第 1 部分 3.3）。消融时关闭 `use_attention`，保持参数与计算路径的 controlled comparison。

### 4.4 训练细节

SGD + cosine lr + warmup；可选 AMP 与梯度裁剪；交叉熵可选 label smoothing。每 batch 重置 LIF 膜电位等状态。

---

## 5 实验设置

**数据集**：PlantVillage（color）；报告类别数、训练/验证或测试划分方式（HF 官方 split 或 Kaggle 本地 `val_ratio`）。

**主模型（本文核心）**：`MSFRes2Net`，`use_msf=True`，`use_attention=True`，**$T=8$**，表 1 其余默认超参。

**对比与消融**：

| 编号 | 配置名称 | 说明 |
|------|----------|------|
| M0 | **Main-MSF-T8** | 主模型：MSF + CSA，$T=8$ |
| M1 | LIF-T8 | `no_msf`，其余同 M0 |
| M2 | MSF-T8-noAttn | `no_attention` |
| M3 | LIF-T8-noAttn | `ablation_baseline` |
| M4 | MSF-T4 / MSF-T16 | 仅改 $T$，观察时延–精度 |
| M5 | MSF-T8-surrogate | 固定 MSF，比较 rect / arctan / sigmoid 等 |

**评价**：Top-1；全局与分层脉冲率；可选宏平均 F1。

**表 1（请填实测）主结果与消融**

| 模型 | Val Top-1 | Test Top-1 | Spike rate (val) | 备注 |
|------|-----------|------------|------------------|------|
| M0 Main-MSF-T8 | — | — | — | 主模型 |
| M1 LIF-T8 | — | — | — | MSF 消融 |
| M2 MSF-T8-noAttn | — | — | — | 注意力消融 |
| M3 baseline | — | — | — | 双关 |
| M4 $T{=}4$ / $16$ | — / — | — / — | — / — | 时间预算 |

---

## 6 结果分析与展示

### 6.1 收敛行为

绘制 train/val loss 与 accuracy；**MSF vs LIF** 若出现明显训练速度差异，可与 surrogate 梯度有效支撑宽度联系讨论。

### 6.2 脉冲活动与稀疏性

报告 `SpikeCounter` 全局率及若干浅层/深层曲线。**主模型 MSF-T8** 应在精度与脉冲率之间取得合理折中；$T$ 增大通常增加每样本前向时间，脉冲统计需在同口径下比较。

### 6.3 消融解读（预期性分析框架，非替真实验下结论）

- **M1 vs M0**：若 MSF 提升精度或加快收敛，可归因于多阈值反向路径对深层梯度的改善；若脉冲率显著变化，可讨论是否与发放正则相关。
- **M2 vs M0**：若 CSA 稳定提升 val，说明病斑与通道重标定在任务上有效；若差距小，可能数据增强已足够或注意力容量需调整。
- **M4**：$T$ 过小可能限制时间信息容量；过大则训练时间与显存上升，收益可能饱和。

### 6.4 可视化建议

输入脉冲栅格（亮度–时间对应）、首层 MSF 膜电位轨迹、混淆矩阵、类间易混样本。

---

## 7 总结与展望

本文实现了面向 PlantVillage 的 **MSF-Res2Net + CSA** 脉冲分类系统，以 **$T=8$ 的 MSF 配置** 为主模型，给出 **LIF 替代、注意力关闭、时间步与 surrogate** 等消融协议，并从精度与脉冲率双维度分析。未来工作包括：（1）更鲁棒的编码（结合颜色归一化与光照增强）；（2）ANN–SNN 知识蒸馏或混合模型；（3）在神经形态数据集格式或硬件约束下的验证；（4）更长训练与 AutoML 式 surrogate 超参搜索。

---

## 参考文献（示例格式，请按正式出版信息核对）

1. Mohanty S. P., *et al.* PlantVillage: Using deep learning for image-based plant disease detection. *Front. Plant Sci.*, 2016.
2. Fang W., *et al.* SpikingJelly: An open-source deep learning framework for spike-based computation. *Patterns*, 2023.
3. Gao S.-H., *et al.* Res2Net: A new multi-scale backbone architecture. *TPAMI*, 2021.
4. Neftci E., *et al.* Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 2019.
5. Fan L., *et al.* Multi-synaptic firing / MSF 相关神经形态学习（请替换为你在 README 中引用的 *Nature Communications* 2025 条目及页码/DOI）。

---

## 附录 A：复现命令摘要

```bash
# 主模型（默认 MSF，T 在代码或参数中设为 8）
python train.py --run_name msf_t8 --epochs 100 --batch_size 32 --lr 0.1

# 消融
python train.py --preset no_msf --run_name lif_t8 --epochs 100 --batch_size 32 --lr 0.1
python train.py --preset no_attention --run_name msf_t8_noattn --epochs 100 --batch_size 32 --lr 0.1

# 测试
python test.py --ckpt checkpoints/best_msf_res2net.pt --batch_size 64
```

Notebook：`notebooks/kaggle_plantvillage_snn.ipynb` 中通过 `get_config(..., T=8, msf_decay=..., ...)` 与 `cfg.*` 赋值对齐上述协议。

---

*文档版本与仓库代码同步维护；数值结果以你实际跑出的日志与 checkpoint 为准。*
