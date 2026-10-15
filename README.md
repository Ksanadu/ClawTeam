# ClawTeam

**项目时间：2026.3 – 2026.5**

> 面向肿瘤多学科会诊（Tumor Board / MDT）的可审计多智能体 Harness 系统。

ClawTeam 不是单纯训练一个医疗大模型，而是围绕肿瘤 MDT 场景搭建一个可观测、可消融、可部署的 **Harness Engineering** 框架。系统通过复杂度评估、专科角色编排、独立意见生成、反驳修正和共识仲裁，把单次黑盒回答转化为可追踪的临床推理流程。

本项目用于课程项目、系统演示和实验评测，不用于真实医疗诊断、处方、手术或放疗决策。

## 项目亮点

- **可追溯**：记录复杂度判断、专科意见、反驳修正、模型后端调用和最终仲裁过程。
- **可审计**：可以检查每个专家为什么提出某个治疗路径，以及分歧如何被解决。
- **可消融**：支持 E1 单 Agent、E2 多角色、E3 多角色 + Round 2、E3B 本地基座、E4 LoRA 混合部署等对比。
- **可部署**：外科和肿瘤内科使用本地 Qwen3-4B LoRA，病理科和放疗科保留 API 专家，形成本地 + 云端的混合架构。
- **贴近 MDT 流程**：输出共识、分歧、条件分支、风险提示和可执行治疗时间线，而不是只给一个最终答案。

## 系统工作流

```text
输入病例
   |
   v
复杂度评估：simple / moderate / complex
   |
   +-- simple   -> 聚焦单 Agent 回答
   |
   +-- moderate -> 四专家 Round 1 独立意见 + 汇总
   |
   +-- complex  -> Round 1 独立意见
                   -> Round 2 反驳与修正
                   -> Coordinator 共识仲裁
   |
   v
结构化 MDT 治疗方案
```

在前端和实验中，E1、E2、E3、E4 等模式可以显式选择；复杂度评估用于控制所选模式中的推理深度。当前实验默认使用固定一轮 Round 2，以保证可复现实验对比。对于真实部署中的高复杂病例，反驳阶段可以扩展为有上限的迭代机制，例如最多 2 到 3 轮，并在专家意见稳定或无重大分歧时提前停止。

## 四个专科角色

| 角色 | 后端 | 主要职责 |
|------|------|----------|
| 外科医生 | 本地 Qwen3-4B + Surgeon LoRA | 可切除性、术式、淋巴结清扫、围手术期风险 |
| 肿瘤内科医生 | 本地 Qwen3-4B + Oncology LoRA | 系统治疗、靶向治疗、免疫/化疗、治疗顺序 |
| 病理科医生 | Qwen3.6-plus API，支持可选视觉输入 | 组织学诊断、分子标志物、分期线索、病理报告或图像解释 |
| 放疗科医生 | Qwen3-Max API | 放疗适应证、剂量分割、OAR 保护、与手术/系统治疗衔接 |

这样分配模型的原因是：外科和肿瘤内科是高频核心治疗决策角色，适合本地 LoRA 专家降低云端依赖；病理科可能需要图像输入，放疗科需要较强的文本推理和安全约束，因此保留 API 专家。

## Harness 核心模块

| 模块 | 说明 |
|------|------|
| Complexity Assessor | 使用规则、正则和 LLM fallback 判断病例复杂度 |
| Specialist Role Orchestration | 编排外科、肿瘤内科、病理科、放疗科四个角色 |
| Round 1 Independent Opinions | 四个专家独立生成初始意见，保留各自视角 |
| Round 2 Rebuttal and Correction | 专家互相审阅意见，输出同意、反对、修正和最终专科建议 |
| Coordinator Arbitration | 类似 Tumor Board 主任，综合证据、安全性、可行性和专家相关性权重生成最终方案 |
| LoRA Plug-in | 外科和内科可切换本地 Qwen3-4B base 或 LoRA 专家 |
| Evaluation Harness | 通过 06 和 07 notebook 进行 MCQ 消融与 TumorBoard 定性评估 |

Coordinator 不是第五个专家。它是共识仲裁模块，会使用角色相关性权重作为辅助参考，但最终输出不是简单加权投票，而是优先考虑临床证据、患者安全、治疗可行性和未解决分歧。

## 实验设置与结果

### LoRA 训练数据

两个本地 LoRA 专家基于 Qwen3-4B 分别训练：

| LoRA 专家 | 训练样本数 | 用途 |
|----------|------------|------|
| 外科 LoRA | 3,779 | 可切除性、术式、围手术期风险 |
| 肿瘤内科 LoRA | 4,238 | 系统治疗、靶向/化疗/免疫、治疗顺序 |

训练数据包含医学问答和病例式指令数据，并通过真实临床题目种子进行 LLM 辅助扩充。评测前通过 leakage filtering 去除与 LoRA 训练数据重叠的 benchmark 病例。

### MCQ 消融实验

定量评估使用 MedQA 和 CMExam 的肿瘤相关 held-out MCQ 病例。原始 benchmark pool 包含 240 个 MedQA oncology test cases 和 768 个 CMExam oncology test cases；过滤训练重叠后剩余 776 个可用公开 MCQ 病例。主实验抽取 150 个病例，其中 MedQA 75 个、CMExam 75 个。

| 方法 | 配置 | 准确率 | 正确数 |
|------|------|--------|--------|
| E1 | 单 Agent DeepSeek API | 0.900 | 135/150 |
| E2 | 四专家 Round 1，全 API | 0.893 | 134/150 |
| E3 | 四专家 + Round 2，全 API | 0.887 | 133/150 |
| E3B | E3 harness，但外科/内科换成本地 Qwen3-4B base | 0.833 | 125/150 |
| E4 | E3B + 外科/内科 LoRA | 0.880 | 132/150 |

主要结论：

- 强单模型 DeepSeek 在 MCQ 上最高，这是合理的，因为选择题偏向简洁单答案推理。
- ClawTeam 不声称在 MCQ 上超过最强单 Agent。
- 严格 LoRA 对比应看 **E4 - E3B**：同一 harness、同一测试集、同一病理/放疗 API，只把外科和内科从本地 base 换成 LoRA，准确率从 0.833 提升到 0.880，增益为 **+4.7 percentage points**。
- 与 E3 全 API debate 相比，E4 用本地 LoRA 替换两个核心专家，专家云端 API 调用约减少 50%，但准确率仍接近 E3。

### TumorBoard 定性评估

为了补充 MCQ 不能体现 MDT 自由文本规划能力的问题，我们额外构建 TumorBoard free-form 病例，并使用 LLM judge 比较 E1 单 Agent 和 E4 ClawTeam。

5 个 TumorBoard cases 的 LLM judge 结果如下：

| 指标 | E1 Single | E4 ClawTeam | 提升 |
|------|-----------|-------------|------|
| Medical accuracy | 4.00 | 4.80 | +0.80 |
| Plan completeness | 4.40 | 4.80 | +0.40 |
| MDT reasoning | 2.60 | 5.00 | +2.40 |
| Actionable timeline | 4.20 | 4.80 | +0.60 |
| Average score | 3.80 | 4.85 | +1.05 |

E4 在 5 个病例中赢下 4 个，最大提升来自 MDT reasoning。这说明 ClawTeam 的价值主要体现在多专科综合、分歧处理和治疗路径构建，而不是单项选择题分数。

LLM judge 不是临床金标准，只作为自由文本 MDT 方案质量的补充评估。后续应引入更多病例和真实临床专家评分。

## 项目结构

```text
ClawTeam/
├─ backend/
│  ├─ api/                         # FastAPI 接口
│  ├─ app.py                       # 后端入口
│  ├─ config/
│  │  ├─ .env.example              # 环境变量模板
│  │  └─ config.py
│  ├─ graph/
│  │  ├─ agent.py                  # 单 Agent / 多 Agent 流式入口
│  │  ├─ coordinator.py            # MDT 协调与仲裁
│  │  ├─ complexity_assessor.py    # 复杂度评估
│  │  ├─ guardian.py               # 可选安全守卫
│  │  └─ roles/                    # 四个专科角色
│  ├─ workspace/
│  │  ├─ AGENTS.md                 # 协作协议
│  │  └─ roles/                    # 角色 prompt
│  ├─ skills/                      # 工具/技能说明
│  └─ eval/
│     ├─ datasets/                 # benchmark 下载与处理
│     ├─ data_generators/          # LoRA 数据构建
│     ├─ inference/                # LoRA / local base 加载
│     ├─ notebooks/                # 训练与评估 notebook
│     └─ results/                  # 实验输出
└─ frontend/
   └─ src/
      ├─ app/
      ├─ components/chat/          # MDT 流程链、角色卡片、工具链展示
      ├─ components/layout/
      └─ lib/                      # API 与状态管理
```

## 主要 Notebook

```text
backend/eval/notebooks/03_train_surgeon_qwen3.ipynb
backend/eval/notebooks/04_train_oncologist_qwen3.ipynb
backend/eval/notebooks/06_ablation_evaluation.ipynb
backend/eval/notebooks/07_tumor_board_qualitative_evaluation.ipynb
```

## 快速启动

### 后端

推荐 Python 3.13（回归 CI 使用同一版本）。从仓库根目录创建隔离环境：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux / Git Bash: source .venv/bin/activate
# Windows Git Bash: source .venv/Scripts/activate
python -m pip install -r backend/requirements.txt
cp backend/config/.env.example backend/config/.env
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8002 --reload
```

`requirements.txt` 是锁定版本的 API 运行依赖；Chroma、Postgres checkpointer、Langfuse 等可选功能使用 `backend/requirements-memory.txt`。训练依赖继续独立放在 `backend/eval/requirements_train.txt`，GPU / CUDA 环境需按设备单独配置。默认模板关闭本地 LoRA，配置模型权重和训练依赖后再显式开启。

本地没有 LoRA 模型时，可以在 `backend/config/.env` 中关闭 LoRA：

```env
USE_LORA_SURGEON=false
USE_LORA_MEDICAL_ONCOLOGIST=false
```

### 前端

```bash
cd frontend
npm ci
npm run dev
```

默认前端端口：

```text
http://localhost:7788
```

默认仅监听本机。前端 API 地址可通过 `NEXT_PUBLIC_API_BASE_URL` 设置（包含 `/api`），修改后需重新启动开发服务器或重新构建。HTTPS 部署建议使用同源 `/api` 反向代理。

不要直接把当前 API 暴露到公网或不可信局域网。远程部署需要认证反向代理、TLS，并显式设置 `CLAWTEAM_ALLOWED_HOSTS` 和 `CLAWTEAM_ALLOWED_ORIGINS`；来源/Host 检查不等于用户认证。单 Agent 的 Shell/Python 工具仍是可信本机能力，不是沙箱。

测试、锁文件更新、会诊状态与部署限制见 [开发与回归测试说明](docs/DEVELOPMENT.md)。已验证的 Linux / Windows CI 均为 **171 项后端测试通过、0 项跳过**，前端检查全部通过；对应提交与日志见 [基线验证结果](docs/BASELINE_VALIDATION.md)。

仓库包含项目源码、训练与评测 notebook、实验结果和演示截图。API 密钥、模型权重、运行时记忆数据库及本地知识库参考资料不随仓库发布；请按环境变量模板配置自己的模型服务和本地资源。

## 环境变量示例

### DeepSeek 默认模型

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 本地 LoRA 专家

```env
USE_LORA_SURGEON=true
LORA_SURGEON_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_SURGEON_PATH=eval/models/surgeon_qwen3_lora

USE_LORA_MEDICAL_ONCOLOGIST=true
LORA_MEDICAL_ONCOLOGIST_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_MEDICAL_ONCOLOGIST_PATH=eval/models/oncologist_qwen3_lora
```

### Qwen / DashScope 角色 API

```env
DASHSCOPE_API_KEY=sk-...
BAILIAN_API_KEY=sk-...

PATHOLOGIST_LLM_PROVIDER=bailian
PATHOLOGIST_LLM_MODEL=qwen3.6-plus
PATHOLOGIST_LLM_API_KEY=sk-...
PATHOLOGIST_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

PATHOLOGIST_VISION_ENABLED=true
PATHOLOGIST_VISION_PROVIDER=bailian
PATHOLOGIST_VISION_MODEL=qwen3.6-plus
PATHOLOGIST_VISION_API_KEY=sk-...
PATHOLOGIST_VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RADIATION_ONCOLOGIST_LLM_PROVIDER=bailian
RADIATION_ONCOLOGIST_LLM_MODEL=qwen3-max
RADIATION_ONCOLOGIST_LLM_API_KEY=sk-...
RADIATION_ONCOLOGIST_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

请不要把真实 API Key 提交到 GitHub。

## 技术栈

### Backend

- Python
- FastAPI + SSE
- LangChain / LangGraph 风格的角色编排
- OpenAI-compatible LLM API
- Transformers / PEFT / Qwen3 LoRA
- 可选 Guardian、安全与记忆模块

### Frontend

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- lucide-react

## 当前状态

已完成：

- 肿瘤 MDT 四专家角色
- 复杂度评估器
- Round 1 独立意见与 Round 2 反驳修正
- Coordinator 共识仲裁
- 外科和肿瘤内科 Qwen3-4B LoRA 训练
- 病理科 Qwen3.6-plus API 与可选视觉输入
- 放疗科 Qwen3-Max API
- 前端 MDT 流程链、工具链聚合、角色意见卡片
- 06 MCQ 消融实验
- 07 TumorBoard 定性评估

仍可继续改进：

- 扩大 TumorBoard free-form benchmark
- 引入真实临床专家评分
- 增强指南、TNM、药典、剂量计算等 evidence-grounded tool use
- 完整评估 Guardian 和安全边界
- 将 Round 2 扩展为有成本上限的多轮迭代机制

## 免责声明

ClawTeam 是课程项目和研究原型，仅用于医学多智能体系统演示与实验评测。系统输出不能替代执业医生的诊断、处方、手术方案、放疗计划或任何真实临床决策。
