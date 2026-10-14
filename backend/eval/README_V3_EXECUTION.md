# 🎯 ClawTeam v3.1 — 执行手册（你照着跑就行）

> 肿瘤多学科会诊（Tumor Board）系统  
> Harness Engineering（70%）+ Qwen3-4B LoRA 训练（30%）  
> 截止：2026-05-07

---

## 📁 已完成的代码改造（无需你做）

### 后端（Harness 框架核心）
- ✅ 4 个肿瘤专科角色 prompt：`workspace/roles/{PATHOLOGIST,SURGEON,MEDICAL_ONCOLOGIST,RADIATION_ONCOLOGIST}.md`
- ✅ 4 个角色类：`graph/roles/{pathologist,surgeon,medical_oncologist,radiation_oncologist}.py`
- ✅ 角色基类支持 Round 1 + Round 2：`graph/roles/base_role.py`
- ✅ 复杂度评估器（替代 Router）：`graph/complexity_assessor.py`
- ✅ 多轮辩论协调器（真协作 Harness）：`graph/coordinator.py`
- ✅ AgentManager 接入 v3 流程：`graph/agent.py`

### 数据 + 训练
- ✅ Benchmark 下载：`eval/datasets/download_{medqa,medbullets,cmexam,pubmedqa}.py`
- ✅ 真实数据准备：`eval/data_generators/prepare_{surgeon,oncologist}_data.py`
- ✅ Qwen3-4B 训练 notebook：`eval/notebooks/03_train_surgeon_qwen3.ipynb`、`04_train_oncologist_qwen3.ipynb`
- ✅ 消融评测 notebook：`eval/notebooks/06_ablation_evaluation.ipynb`

### 已归档（保留对比基线）
- 📦 `eval/_archived/`：旧角色、旧数据生成、旧训练 notebook（v0.1 MVP 文物）

---

## 🚀 你接下来要做的（按顺序）

### Step 1: 租 AutoDL 4090（5 分钟）

1. 登录 https://www.autodl.com
2. 选 RTX 4090（24GB），按量计费
3. 镜像：PyTorch 2.4+ / CUDA 12.4+
4. SSH 连接

### Step 2: 拉项目 + 装依赖（10 分钟）

```bash
# 在 AutoDL 服务器
cd /root/autodl-tmp
git clone https://github.com/Miles2333/ClowTeam_NLP.git
cd ClowTeam_NLP/backend
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 训练专用依赖
pip install peft trl accelerate datasets jupyterlab transformers \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.7.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
```

### Step 3: 配置 .env（2 分钟）

```bash
cp config/.env.example config/.env
# 编辑 config/.env，填入：
# LLM_PROVIDER=deepseek
# LLM_API_KEY=sk-xxx
# LLM_BASE_URL=https://api.deepseek.com
```

### Step 4: 下载 Benchmark（30-60 分钟）

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
export HF_ENDPOINT=https://hf-mirror.com

python eval/datasets/download_medqa.py
python eval/datasets/download_cmexam.py
python eval/datasets/download_pubmedqa.py
python eval/datasets/download_medbullets.py  # 失败也没事
```

数据会自动筛选肿瘤子集到 `eval/datasets/data/oncology/`。

### Step 5: 准备训练数据（1-2 小时，调 DeepSeek API）

```bash
python eval/data_generators/prepare_surgeon_data.py     # 真实数据 + 400 条扩充
python eval/data_generators/prepare_oncologist_data.py  # 真实数据 + 400 条扩充
```

API 成本约 ¥10-20。完成后看 `eval/datasets/data/training/`。

### Step 6: 启动 JupyterLab（1 分钟）

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

在 AutoDL 控制台开端口转发，本地浏览器打开。

### Step 7: 训练 LoRA（4-6 小时）

按顺序跑：
1. **`03_train_surgeon_qwen3.ipynb`** — 外科 LoRA（2-3 h）
2. **`04_train_oncologist_qwen3.ipynb`** — 内科 LoRA（2-3 h）

每个 notebook：Restart Kernel → 从头跑到尾。

### Step 8: 配置 LoRA 启用（1 分钟）

编辑 `config/.env`，加：

```env
USE_LORA_SURGEON=true
LORA_SURGEON_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_SURGEON_PATH=eval/models/surgeon_qwen3_lora

USE_LORA_MEDICAL_ONCOLOGIST=true
LORA_MEDICAL_ONCOLOGIST_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_MEDICAL_ONCOLOGIST_PATH=eval/models/oncologist_qwen3_lora

# Guardian（可选；只有确认 guardian_bert 存在并已接入时再打开）
USE_TRAINED_GUARDIAN=false
TRAINED_GUARDIAN_PATH=eval/models/guardian_bert
```

### Step 9: 跑消融实验（2-3 小时）

打开 **`06_ablation_evaluation.ipynb`**，按顺序跑所有 cell。

输出文件：
- `eval/results/main_ablation.csv`（论文 Table 1）
- `eval/results/main_ablation.png`（论文 Figure 1）
- `eval/results/round_comparison.csv`
- `eval/results/final_report.json`

### Step 10: 端到端 Demo（验证真协作）

```bash
cd /root/autodl-tmp/ClowTeam_NLP/backend
uvicorn app:app --host 0.0.0.0 --port 8002
```

在 AutoDL 开 8002 端口。本地前端：

```bash
cd frontend
npm install
npm run dev
```

修改 `frontend/src/lib/api.ts` 的 `getApiBase()` 指向 AutoDL 公网 URL。

---

## 📊 论文里要写的关键指标

| 指标 | 来源 | 期望值 |
|------|------|------|
| 主实验 4 组准确率 | E1 → E4 阶梯式上升 | E4 > E3 > E2 > E1 |
| Round 2 修正率 | `06_ablation_evaluation.ipynb` 计算 | > 30% 说明真协作 |
| 训练 vs 不训练提升 | E4 vs E3 | +3-8% |
| Harness 价值 | E3 vs E2 | +5-10% |
| 整体提升 | E4 vs E1 | +10-15% |

---

## ⚠️ 风险点 + 应对

| 风险 | 应对 |
|------|------|
| Qwen3-4B 显存不够 | 改 `MODEL_NAME = 'Qwen/Qwen3-1.7B-Instruct'` |
| MedBullets 下载失败 | 跳过，用 MedQA + CMExam 已经够 |
| CMExam 下载失败 | 用 MedQA 英文题，论文里说明 |
| 消融实验 acc 没有阶梯式上升 | 检查 Coordinator 的 `force_complexity` 参数是否生效；增大 test_cases 到 200 |
| Round 2 修正率 < 10% | prompt 不够强：增加"必须给出反驳"的强度 |

---

## 👥 分工建议

| 同学 | 任务 |
|------|------|
| **A（你）** | Harness 整合（已完成）+ 调试端到端 Demo + 演示 |
| **B** | Step 4-5（下载 + 数据准备）+ Step 9（跑消融）+ 出图 |
| **C** | Step 7（训练 2 个 LoRA）+ 训练 vs 不训练 ablation |

---

## 📅 7 天时间表

| 日期 | 任务 |
|------|------|
| **5/01** | 租 AutoDL + 装依赖 + 下载 Benchmark + 准备训练数据 |
| **5/02** | 训练外科 LoRA（4 小时）+ 整理 Tumor Board 案例集 |
| **5/03** | 训练内科 LoRA（4 小时）+ 调试端到端 Demo |
| **5/04** | 跑主实验 E1-E4（约 4-6 小时 API 时间）|
| **5/05** | 跑子消融 + 出图 + 整理实验数据 |
| **5/06** | 写论文 + 做 PPT |
| **5/07** | 提交 |

---

## 💡 论文亮点（写报告时强调）

1. **中文 Tumor Board MDT 系统**（中文 MDT 论文稀缺）
2. **真协作 Harness**：Round 2 强制反驳设计 + 修正率指标
3. **Harness × LoRA 混合架构**：四象限消融对比
4. **复杂度自适应**：简单题省 token、复杂题保性能
5. **基于公开 Benchmark 真实数据训练**（非纯合成）

---

## ✅ TA 反馈对照（写报告时引用）

| TA 反馈 | 本项目落实 |
|--------|---------|
| Benchmark 先行 | MedQA/MedBullets/CMExam 肿瘤子集 |
| Harness 为主 | 70% 工作量在多轮辩论 + 共识 |
| Router → 角色分工 | 改名"复杂度评估器"，4 角色都参与 |
| 共享记忆放选做 | 主线不依赖，时间够再做 |
| 消融实验 | 4 主 + 3 子，训练前后 + Harness 前后都对比 |
| 真协作（不是流水线）| Round 2 强制反驳，量化修正率 |
| 用强基座 | Qwen3-4B（云端 4090）|

---

完整方案落地完毕。**你下一步**：把这份手册转给组员，开始 Step 1。
