# 推理加载工具

本目录用于把训练好的模型接入 ClawTeam 后端。当前 v3.1 主线是肿瘤 MDT：

- 病理科：API + prompt
- 肿瘤外科：Qwen3-4B LoRA（可选）
- 肿瘤内科：Qwen3-4B LoRA（可选）
- 放疗科：API + prompt
- Guardian：可选 BERT 安全守卫

所有切换都通过环境变量完成，不需要改主流程代码。

## 当前加载器

| 模块 | 替换位置 | 触发开关 |
|------|----------|----------|
| `load_guardian.py` | Guardian 安全判断 | `USE_TRAINED_GUARDIAN=true` |
| `load_lora_role.py` | `graph/roles/base_role.py` 中外科/内科角色 LLM 调用 | `USE_LORA_SURGEON=true` / `USE_LORA_MEDICAL_ONCOLOGIST=true` |

旧的 router/physician/pharmacist/radiologist 路线属于早期 MVP，不是当前 Tumor Board 主线。

## AutoDL .env 示例

在云端 `backend/config/.env` 中加入：

```env
USE_LORA_SURGEON=true
LORA_SURGEON_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_SURGEON_PATH=eval/models/surgeon_qwen3_lora

USE_LORA_MEDICAL_ONCOLOGIST=true
LORA_MEDICAL_ONCOLOGIST_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_MEDICAL_ONCOLOGIST_PATH=eval/models/oncologist_qwen3_lora

# Optional Guardian
USE_TRAINED_GUARDIAN=false
TRAINED_GUARDIAN_PATH=eval/models/guardian_bert
```

本地 Windows 如果没有这两个 LoRA 目录，应保持 `USE_LORA_* = false`，否则会回退到 API + prompt。

## 实验切换

| 实验场景 | 配置 |
|----------|------|
| API + prompt 基线 | `USE_LORA_SURGEON=false`，`USE_LORA_MEDICAL_ONCOLOGIST=false` |
| 外科 LoRA | 只开启 `USE_LORA_SURGEON=true` |
| 内科 LoRA | 只开启 `USE_LORA_MEDICAL_ONCOLOGIST=true` |
| 完整 LoRA 方案 | 外科和内科都开启 |
| Guardian 子消融 | 单独切换 `USE_TRAINED_GUARDIAN` |

每次切换 `.env` 后需要重启后端。

## 快速验证

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
python - <<'PY'
from eval.inference.load_lora_role import load_lora_role
for role in ["surgeon", "medical_oncologist", "pathologist", "radiation_oncologist"]:
    agent = load_lora_role(role)
    print(role, "LoRA" if agent else "API/prompt fallback")
PY
```
