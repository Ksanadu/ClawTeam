# 归档目录

ClawTeam v3.1 升级时归档的文件。**这些文件已不在主线使用**，但保留作为：
1. 历史参考（v0.1 MVP 的实现）
2. 论文里的对比基线（如 0.5B 主治 LoRA 作为"小模型基线"）

## 归档内容

### 旧角色（v0.1 通用 agent）

- `physician.py` / `pharmacist.py` / `radiologist.py` — 已被 4 个肿瘤专科取代
  - 新角色：`graph/roles/{pathologist,surgeon,medical_oncologist,radiation_oncologist}.py`

### 旧数据生成脚本

- `gen_physician_data.py` — 纯 LLM 生成（已被 `prepare_surgeon_data.py` 取代，新版用真实 Benchmark + LLM 扩充）
- `gen_pharmacist_data.py` — 已废弃（药师不再是 Tumor Board 角色）
- `gen_radiologist_data.py` — 已废弃（影像/放疗调 API）

### 旧训练 notebook

- `03_train_physician_lora.ipynb` — Qwen2.5-0.5B 主治（保留训练好的模型作为"小模型基线"）
- `04_train_pharmacist_lora.ipynb` — Qwen2.5-0.5B 药师
- `05_train_radiologist_vl.ipynb` — 多模态 VL（按 TA 反馈，影像调 API 即可）

## 替代方案

| 旧文件 | 新文件 |
|------|------|
| `physician.py` | `graph/roles/pathologist.py` 等 4 个 |
| `gen_physician_data.py` | `eval/data_generators/prepare_surgeon_data.py` |
| `03_train_physician_lora.ipynb` | `eval/notebooks/03_train_surgeon_qwen3.ipynb` |
| `04_train_pharmacist_lora.ipynb` | `eval/notebooks/04_train_oncologist_qwen3.ipynb` |
| `05_train_radiologist_vl.ipynb` | 直接调多模态 API（如 Qwen-VL API） |
