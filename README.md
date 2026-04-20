# coder

用 Python 从零实现的 Code Agent，架构参考 Claude Code 源码逆向分析。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API
cp .env.example .env
# 编辑 .env，填入你的 key

# 启动
python main.py

# 恢复上次会话
python main.py --resume

# 列出历史会话
python main.py --list
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ANTHROPIC_API_KEY` | API key | `sk-...` |
| `ANTHROPIC_BASE_URL` | API 地址（默认 Anthropic，可换兼容的模型例如：DeepSeek）| `https://api.deepseek.com/anthropic` |
| `MODEL_ID` | 模型名称 | `deepseek-chat` / `claude-opus-4-6` |

## 功能

- **流式 Agent 主循环**：SSE 实时解析，tool_use 块并发执行，速率限制 + 上下文超限自动恢复
- **7 个内置工具**：Bash / Read / Edit / Write / Glob / Grep / Agent（子 agent 分派）
- **子 Agent 系统**：三种预设 —— `general-purpose`（通用，带 worktree 隔离）/ `Explore`（只读快速搜索）/ `Plan`（只读架构设计）。每次 dispatch 拿独立 system prompt 和 scoped tool list，UI 实时显示子 agent 进度
- **Worktree 隔离**：写权限子 agent 在独立 git worktree + 新分支上工作，跑完干净就清理，有改动就保留路径 + 分支名供 review，不污染主工作区
- **六层权限系统**：hard-deny → settings deny → read-only → allow rules → session → ask，规则支持 `Bash(git add *)` glob 匹配，persist 写入 `.coder/settings.json` 跨会话复用
- **Hooks 系统**：`PreToolUse` / `PostToolUse` 钩子，支持 shell 命令 + 回调；内置 `dangerous_bash_guard` 和 `file_write_audit`
- **LLM 摘要压缩**：监控 token 占比触发 auto-compact，历史对话压缩为结构化摘要
- **工具结果持久化**：大输出自动 offload 到 `~/.coder/sessions/<proj>/<session>.tool-results/`，消息里只留 preview + 文件路径，节省 token
- **会话持久化**：JSONL 存储，支持 `--resume` 断点续话
- **环境感知**：系统提示词注入 cwd、git 状态与分支、OS、Shell、当前日期

## 架构

```
main.py                  # CLI 入口，REPL 循环，slash 命令
agent_loop.py            # Agent 主循环：call_llm → exec_tools → repeat
context.py               # 对话历史 + 系统提示词构建（含环境信息）
streaming_executor.py    # 流式工具执行，concurrent-safe 并发调度
session.py               # 会话持久化（JSONL 存储 + resume，懒 mkdir）
settings.py              # 两层配置加载/合并/更新（user + project）
agent_types.py           # ToolResult, ToolUseBlock, StreamEvent 数据类型

permissions/
  manager.py             # 六层权限检查
  rules.py               # 权限规则解析与匹配（ToolName(pattern) 格式）

hooks/
  runner.py              # HookRunner：跑 shell command + callback hooks
  builtin.py             # 内置 hook：dangerous_bash_guard + file_write_audit

subagents/
  registry.py            # AgentDefinition + 注册表
  general_purpose.py     # 通用子 agent（worktree 隔离）
  explore.py             # 只读搜索子 agent
  plan.py                # 只读架构设计子 agent

services/
  compact.py             # LLM 摘要压缩 + auto-compact
  tool_result_storage.py # 大工具结果 offload 到磁盘
  worktree.py            # git worktree 创建 / 清理 / 变更检测

tools/
  base.py                # Tool 抽象基类
  bash.py                # 执行 shell 命令
  file_read.py           # 读文件
  file_edit.py           # 精确字符串替换
  file_write.py          # 写文件
  glob_tool.py           # 文件匹配
  grep_tool.py           # 正则搜索
  agent_tool.py          # 子 agent 分派
```

## Slash 命令

在 REPL 里可用：

| 命令 | 说明 |
|------|------|
| `/compact` | 手动触发 LLM 摘要压缩 |
| `/tokens` | 显示当前估算 token 数 |
| `/sessions` | 列出本项目的历史会话 |
| `/clear` | 清空当前对话历史 |
| `/help` | 显示命令列表 |

## 运行测试

```bash
python -m pytest tests/ -v
```
