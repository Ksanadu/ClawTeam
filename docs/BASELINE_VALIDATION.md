# 最低工程基线验证结果

## 已完成

- 文件接口规范化后白名单、会话 UUID4 校验、缺失会话 404、原子 JSON 写入。
- 有界多轮上下文、重启后的 checkpoint 恢复、压缩摘要生效、单/多 Agent 切换上下文一致。
- 专家失败隔离、专家/轮次/阶段超时、断开后的任务取消与等待、同步本地推理有界执行。
- 工具 `tool_call_id` 端到端关联、乱序结果保存、失败/停止/空输出成功状态显示。
- 完整会诊元数据和附件保存/回放、快速会话切换隔离、流式中断和停止生成。
- 蒸馏失败重试、成功幂等、同会话 advisory lock、非破坏性 schema 初始化。
- 核心/开发/可选记忆依赖锁、前端锁文件、Windows/Linux CI、开发说明与本机默认配置。

## 本地执行结果

| 检查 | 结果 |
|---|---|
| 后端 pytest（Python 3.13.12） | 162 passed，9 skipped，1 dependency deprecation warning |
| 前端 Vitest | 31 passed，5 test files |
| TypeScript `typecheck` | 通过 |
| ESLint | 通过，无 warning/error |
| Next.js 14.2.35 production build | 通过 |
| `npm ci` | 通过 |
| `pip check` | 通过，无依赖冲突 |
| `git diff --check` | 通过，仅 Windows LF/CRLF 提示 |

9 项跳过均因当前 Windows 不允许测试进程创建符号链接（WinError 1314）；模拟规范化重定向检查仍执行。Linux 测试由 GitHub Actions 执行，结果以对应提交的工作流记录为准。

## 验证边界与剩余限制

- 测试使用临时文件、真实模块与干净的模型/仓储替身，未调用真实模型、生产数据库或真实病例数据。
- 未执行浏览器 GUI 冒烟、GPU 性能/强制终止、真实 PostgreSQL/pgvector 锁与迁移验证。
- 进程内会话互斥不支持多个 API worker；运行中的原生/GPU生成不能被 Python 请求取消强制终止，但会继续占用唯一推理许可直到结束。
- Next.js 仅做同大版本补丁升级。生产依赖审计仍有4个受影响包（2 low、1 high、1 critical）；完整开发依赖审计17项。不能将此状态称为可安全公网部署，详见 DEVELOPMENT.md。
- 未迁移 Next/React 大版本、引入认证系统/工具沙箱、替换会话数据库、调整研究实验定义或新增临床证据检索。
- Git 提交、推送和远程 CI 结果以仓库历史及 GitHub Actions 为准。
