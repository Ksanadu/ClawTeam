# 开发与回归测试

## 环境与依赖

API / 自动化测试以 Python 3.13 为基准；前端 CI 使用 Node.js 22。GPU 训练与本地 LoRA 是可选环境，不要把训练包直接装入 API 测试环境。

从仓库根目录执行：

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Git Bash: source .venv/Scripts/activate
python -m pip install -r backend/requirements-dev.txt
python -m pip check
python -m pytest
```

- `backend/requirements.in` / `requirements.txt`：核心 API、API 专家和当前运行代码会导入的轻量记忆依赖。
- `backend/requirements-dev.in` / `requirements-dev.txt`：与核心版本对齐的 pytest、异步测试、网络限制与锁文件工具。
- `backend/requirements-memory.in` / `requirements-memory.txt`：可选 Chroma、Postgres checkpointer、DashScope、Langfuse；只在启用对应功能时安装。
- `backend/eval/requirements_train.txt`：原有 GPU/训练/Notebook 依赖，与 API 分开。PyTorch/CUDA 需要按硬件验证，不属于无模型回归测试的环境保证。
- `frontend/package-lock.json`：前端运行与测试依赖，使用 `npm ci` 而非无锁安装。

锁文件完整固定 Python 包版本，不能手动只改生成的 `.txt`。在 Python 3.13 环境中更新 `.in` 后执行：

```bash
python -m piptools compile --strip-extras --output-file backend/requirements.txt backend/requirements.in
python -m piptools compile --strip-extras --constraint backend/requirements.txt --output-file backend/requirements-dev.txt backend/requirements-dev.in
python -m piptools compile --strip-extras --constraint backend/requirements.txt --output-file backend/requirements-memory.txt backend/requirements-memory.in
```

生成器按平台解析依赖。CI 分别安装 Windows/Linux 核心与开发锁文件；可选 GPU、Chroma 与真实 Postgres 集成需在目标环境另行验证。

## 前端检查

```bash
cd frontend
npm ci
npm test
npm run typecheck
npm run lint
npm run build
```

测试使用模拟 fetch、事件流与临时会话数据，不需要启动后端。构建和测试不会请求真实模型。

## 最小回归范围

| 范围 | 覆盖行为 |
|---|---|
| 文件与会话路径 | 规范化后白名单、分隔符/绝对路径/符号链接逃逸、非法 ID、不存在会话的 HTTP 错误 |
| 会话上下文 | 摘要和近期消息进入模型、已有 checkpoint 不重复注入、重建 checkpoint、压缩与生成互斥 |
| 专家生命周期 | 失败隔离、阶段/专家超时、请求取消后清理子任务、失败/部分完成状态 |
| 工具关联 | 以 `tool_call_id` 匹配并行开始/结束，保存与回放不依赖完成顺序 |
| 前端状态与流 | CRLF/LF、任意分块与 UTF-8、未完成 EOF、停止生成、快速切换会话、历史元数据回放 |
| 记忆蒸馏 | 推理/保存失败可重试、成功幂等、强制重做、同会话并发互斥、锁释放 |

pytest 在收集阶段屏蔽开发者 `.env` 和模型凭据；默认禁止外部 socket 连接，允许回环连接和 Unix socket，仅用于 Windows asyncio 的内部唤醒管道等测试运行设施。ASGI 测试直接运行内存应用，不监听端口。真实数据库、模型或医疗数据不应出现在单元回归测试里。

## 会诊记录与执行状态

历史仍兼容旧的 `role`、`content`、`tool_calls`；新消息具有稳定 `id`，并按需保存 `run_id`、实验模式、角色轮次意见、路由、证据、进度、附件、安全拦截和协作指标。工具事件使用同一 `tool_call_id` 关联。

- `done` 表示运行协议完成，`status` 区分成功、部分完成、失败和拦截，不能只凭流结束判断成功。
- `role_error` 是可降级的单专家失败；`error` 是终止错误。
- 无终态事件的 EOF 是中断，前端保留状态并显示错误。
- 同会话的生成、压缩和删除互斥；冲突返回 409。
- 会话 JSON 写入为原子替换，进程内修改使用锁。此最低实现面向单个 API worker；多进程部署必须先使用共享事务存储与分布式互斥，不能直接增加 worker 数。

上下文默认限制为最近 20 条消息、24000 字符；摘要优先保留。`AGENT_HISTORY_MAX_MESSAGES`、`AGENT_HISTORY_MAX_CHARS` 可调整上限。专家、轮次、阶段超时分别使用 `MDT_ROLE_TIMEOUT_SECONDS`、`MDT_ROUND_TIMEOUT_SECONDS`、`MDT_STAGE_TIMEOUT_SECONDS`。

本地推理使用有界 worker，取消请求不会强制杀死已经进入 GPU/原生库的生成；资源许可持续占用到该调用真正结束，避免下一请求与未结束的生成并发。硬终止与 GPU 服务隔离需要独立推理进程/服务。

蒸馏以已保存对象作为成功标记；失败 exchange 下次仍可重试。按 session 的 PostgreSQL advisory lock 要求会话固定连接，**不能使用 transaction-pooling 模式的 PgBouncer**。自动重试需再次触发该会话的蒸馏调用；当前不是持久任务队列。缺失 embedding 保留为 NULL，可通过原有无向量对象查询安排回填。schema 初始化只做非破坏性创建；不兼容的已有表需要显式迁移，不能通过自动 DROP 重建。

## 本机启动与部署边界

后端从 `backend` 启动，默认只绑定 `127.0.0.1:8002`。前端本机开发端口为 7788。`.env.example` 关闭 LoRA 与长期记忆；配置真实服务前先完成离线测试。

- `CLAWTEAM_ALLOWED_HOSTS`：逗号分隔的明确 Host 默认 localhost/回环地址。
- `CLAWTEAM_ALLOWED_ORIGINS`：逗号分隔的可信前端 Origin，必须包含协议和端口，不允许 `*`。
- `NEXT_PUBLIC_API_BASE_URL`：包含 `/api` 的前端 API URL；HTTPS 默认同源，可显式使用 `/api`。修改后重新启动或构建前端。

Host/Origin 限制不是认证。远程访问必须经 TLS 和认证反向代理，并限制直接访问后端端口。Shell/Python 工具仍按服务账户权限运行，当前不提供沙箱；不要向不可信用户公开，也不要认为 Guardian 等同于执行权限控制。医学输出仅用于科研与演示，不用于诊疗决策。

## 依赖安全状态

本次将 Next.js / eslint-config-next 更新至兼容的 14.2.35，并应用无需 `--force` 的兼容依赖修复。版本锁定不等于无漏洞：本地 `npm audit --omit=dev` 仍报告 4 个受影响包（2 low、1 high、1 critical），主要高危项来自 Next.js 14 及其内置 PostCSS；完整开发依赖审计还存在测试/开发工具相关告警。审计结果随安全数据库变化。

当前不应将 Next.js 14 服务直接暴露给不可信网络。彻底处理需要单独规划受支持 Next.js 大版本升级、相关 React/ESLint 兼容性与完整回归；本次没有使用 `npm audit fix --force` 隐式迁移框架。此限制不因单元测试、构建通过而消失。

## CI

`.github/workflows/ci.yml` 在 push/PR 上运行 Windows/Linux 后端回归，以及前端测试、类型检查、lint、生产构建。工作流不包含密钥和真实模型调用。新增文件只会在提交并推送后触发 GitHub Actions；本地验证不代表远程 CI 已运行。
