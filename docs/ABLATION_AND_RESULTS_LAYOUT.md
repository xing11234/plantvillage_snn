# 消融实验规划与 `snn_out` / `ann_out` 结果目录对照

本文档说明：**实验因子如何组合**、**仓库内预设与开关**、以及你当前 **`snn_out/`**、**`ann_out/`** 下子目录建议对应关系（按文件夹名归纳；若某次跑未改名，以 checkpoint 内 `cfg` 字典为准）。

---

## 1. 实验因子总览

| 因子 | 取值 | 说明 |
|------|------|------|
| **范式** | SNN / ANN | ANN 为传统 CNN，结果通常在 `ann_out`；SNN 在 `snn_out`。 |
| **骨干** | MSF-Res2Net / SpikingJelly ResNet-18 | Res2Net：`models/res2net_msf.py`；RN18：`models/spiking_resnet18_backbone.py`。 |
| **神经元（RN18）** | MSF / LIF / **PLIF** | `use_msf=True` → MSF；`use_msf=False` 且 `lif_variant="lif"` → LIF；`lif_variant="plif"` → `ParametricLIFNode`（可学习膜时常）。 |
| **神经元（Res2Net）** | MSF / LIF | `use_msf`；无 PLIF 路径（若需可对 `make_spiking_node` 扩展）。 |
| **注意力（仅 Res2Net）** | CSA 开 / 关 | `use_attention`；`no_attention` preset。 |
| **时间步** | `T`（如 4、8） | 与 `first_spike_coding` 长度一致。 |
| **MSF 代理梯度** | rect / arctan / sigmoid / … | `msf_surrogate`、`msf_surrogate_alpha`。 |
| **优化** | `lr`、`epochs`、`label_smoothing` 等 | 与 `TrainConfig` / Notebook 一致。 |

**主论文模型（约定）**：**MSF-Res2Net，`T=8`**（或你在文中固定的主配置）；ResNet-18 系列作为 **轻量骨干 + 神经元消融**（MSF vs LIF vs PLIF）。

---

## 2. 标准消融矩阵（建议论文表格行）

### 2.1 SNN（ResNet-18 骨干）— 控制 `BACKBONE=sj_resnet18` 与 `RN18_NEURON`

| ID | 名称 | 配置要点 |
|----|------|----------|
| R18-MSF | RN18 + MSF | `RN18_NEURON="msf"`，`use_msf=True` |
| R18-LIF | RN18 + LIF | `RN18_NEURON="lif"`，`lif_variant="lif"` |
| R18-PLIF | RN18 + PLIF | `RN18_NEURON="plif"`，`lif_variant="plif"`，`lif_tau`→`init_tau` 与 LIF 对齐 |

三者保持 **相同 `T`、`lr`、batch、数据与编码`**，仅神经元不同 → 公平对比。

### 2.2 SNN（Res2Net 骨干）— `BACKBONE=res2net`

| ID | 名称 | 配置要点 |
|----|------|----------|
| R2N-Main | MSF + CSA | 默认 `TrainConfig` |
| R2N-noMSF | LIF 替代 MSF | `--preset no_msf` |
| R2N-noAttn | 去 CSA | `--preset no_attention` |
| R2N-Baseline | LIF + 无 CSA | `--preset ablation_baseline` |

### 2.3 ANN 基线

| ID | 名称 | 说明 |
|----|------|------|
| ANN-RN18 | 传统 ResNet-18 | 与 `kaggle_plantvillage_ann.ipynb` / ANN 脚本一致，结果放 `ann_out`。 |

### 2.4 可选扩展消融

- **不同 `T`**：4 vs 8 vs 16（时延–精度）。  
- **不同 `lr`**：如 0.05 vs 0.1。  
- **不同 surrogate**（仅 MSF）：rect vs arctan vs sigmoid。

---

## 3. 与你现有 `snn_out/` 子目录的对应（归纳）

下列根据**文件夹命名**推断实验内容；正式写论文请以 **TensorBoard / `cfg` 快照** 为准。

| 目录（示例） | 推断消融含义 |
|----------------|-------------|
| `Resnet18_MSF_rect_T8_lr0.05` | RN18 + MSF，`T=8`，rect，`lr=0.05` |
| `Resnet18_MSF_rect_T4_lr0.05` / `..._lr0.1` | 同上，`T=4` 或不同 `lr` |
| `Resnet18_MSF_arctan_T4_lr0.05` | MSF + arctan surrogate |
| `Resnet18_MSF_sigmoid_T4_lr0.05` | MSF + sigmoid surrogate |
| `Resnet18_LIF_rect_T4_lr0.05` | RN18 + LIF（`T=4`） |
| `Res2net_MSF_rect_T4_lr0.1` | **Res2Net** + MSF（注意与 Resnet18 区分） |
| `lr_0.05_T8`、`lr_0.1_T4`、`lr_0.05_T4` | 可能为早期命名；对齐需看 events 内 tag 或 ckpt |
| `epoch_20`、`sigmoid_T4`、`arctan_T4` | 短名实验；建议后续统一 `Resnet18_...` 或 `Res2net_...` 前缀 |
| `snn_fig/spike/*` | 论文插图导出（非单次训练根目录） |

**新增 PLIF 跑分后建议目录名**：例如 `Resnet18_PLIF_T8_lr0.05` 或 `Resnet18_LIFvariant_plif_T8_lr0.05`，与现有 `Resnet18_LIF_*` 并列。

---

## 4. `ann_out/` 对照

| 文件/目录 | 推断含义 |
|-----------|----------|
| `best_ann_tv_resnet18.pt` | ANN ResNet-18 最佳权重 |
| `events.out.tfevents.*` | ANN 训练 TensorBoard 日志 |

ANN 与 SNN 的 **数据增强、划分、epoch、lr** 应对齐后再写「ANN vs SNN」结论。

---

## 5. 你「一共有哪些消融」— 清单小结

按**论文叙事**可归纳为 **5 类**（子实验可多篇/附录展开）：

1. **范式**：ANN vs SNN。  
2. **骨干**：Res2Net（主） vs ResNet-18（辅）。  
3. **SNN 神经元（RN18）**：MSF vs LIF vs **PLIF**（新增）。  
4. **结构件（Res2Net）**：MSF on/off；CSA on/off。  
5. **时间与训练**：`T`；`lr`；`msf_surrogate`（仅 MSF）。

若只数「独立对照实验行」，把 **(2)×(3) 的部分与 (4)(5) 交叉** 会膨胀；正文建议 **主表 4–6 行**（Main + 最关键 3–4 个消融），其余放附录或补充材料。

---

## 6. 配置 API 速查

```python
# ResNet-18 + PLIF（与 LIF 同 lif_tau 初值）
cfg = get_config(use_msf=False, lif_variant="plif", T=8, lr=0.05, ...)

# 或 preset
cfg = get_config("plif_rn18", T=8, lr=0.05, ...)
```

Kaggle Notebook：`RN18_NEURON = "plif"` 即可（见 `kaggle_plantvillage_snn.ipynb`）。
