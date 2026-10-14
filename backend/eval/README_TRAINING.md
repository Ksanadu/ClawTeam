# ClawTeam v3.1 训练指南

当前训练主线只训练两个最关键角色：

| 编号 | 任务 | 模型 | 位置 |
|------|------|------|------|
| 03 | 肿瘤外科 LoRA | Qwen3-4B-Instruct | `notebooks/03_train_surgeon_qwen3.ipynb` |
| 04 | 肿瘤内科 LoRA | Qwen3-4B-Instruct | `notebooks/04_train_oncologist_qwen3.ipynb` |

病理科和放疗科不训练，默认使用 API + prompt。旧的主治医生、临床药师、影像科 LoRA 是早期 MVP 路线，已归档，不属于当前 Tumor Board 主线。

## AutoDL 环境

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install peft trl accelerate datasets jupyterlab transformers \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

建议基座模型提前放在：

```text
/root/autodl-tmp/Qwen3-4B-Instruct-2507
```

## 数据准备

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
python eval/data_generators/prepare_surgeon_data.py
python eval/data_generators/prepare_oncologist_data.py
```

训练数据会写入 `eval/datasets/data/training/`。

## 训练顺序

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

按顺序运行：

1. `notebooks/03_train_surgeon_qwen3.ipynb`
2. `notebooks/04_train_oncologist_qwen3.ipynb`

每个 notebook 建议 Restart Kernel 后从头跑到尾。

## 训练输出

训练完成后应存在：

```text
eval/models/surgeon_qwen3_lora/
eval/models/oncologist_qwen3_lora/
```

## 接入后端

在 `backend/config/.env` 中加入：

```env
USE_LORA_SURGEON=true
LORA_SURGEON_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_SURGEON_PATH=eval/models/surgeon_qwen3_lora

USE_LORA_MEDICAL_ONCOLOGIST=true
LORA_MEDICAL_ONCOLOGIST_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_MEDICAL_ONCOLOGIST_PATH=eval/models/oncologist_qwen3_lora
```

然后重启后端：

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
uvicorn app:app --host 0.0.0.0 --port 8002
```

## 消融实验

训练完成并配置 `.env` 后，运行：

```text
notebooks/06_ablation_evaluation.ipynb
```

主实验建议保留四组：

| 组别 | 配置 | 验证点 |
|------|------|--------|
| E1 | 单 Agent API | 基线 |
| E2 | 4 角色独立，无辩论，无 LoRA | 角色分工价值 |
| E3 | 4 角色 + 多轮辩论，无 LoRA | Harness 协作价值 |
| E4 | E3 + 外科/内科 LoRA | 训练增益 |

Guardian 如果已经真正接入并验证，可以作为子消融单独报告。
