# coder

用 Python 从零实现的 Code Agent，架构参考 Claude Code 源码逆向分析。

## 快速开始

```bash
# 安装（推荐 pipx，全局可用）
pipx install .

# 或普通 pip
pip install -e .

# 配置 API key（任选）
export ANTHROPIC_API_KEY=sk-...
# 或写入项目根目录的 .env 文件

# 启动
coder

# 恢复上次会话
coder --resume

# 列出历史会话
coder --list
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ANTHROPIC_API_KEY` | API key | `sk-...` |
| `ANTHROPIC_BASE_URL` | 兼容端点（不填默认 Anthropic） | `https://api.deepseek.com/anthropic` |
| `MODEL_ID` | 模型名称（不填默认 `claude-opus-4-5`） | `deepseek-chat` / `claude-opus-4-6` |
| `CODER_LOG_LEVEL` | 日志级别（默认 `WARNING`） | `DEBUG` / `INFO` |

## 功能

- **流式 Agent 主循环**：SSE 实时解析，tool_use 块并发执行，速率限制 + 上下文超限自动恢复
- **7 个内置工具**：Bash / Read / Edit / Write / Glob / Grep / Agent（子 agent 分派）
- **子 Agent 系统**：三种预设 —— `general-purpose`（通用，带 worktree 隔离）/ `Explore`（只读快速搜索）/ `Plan`（只读架构设计）
- **Worktree 隔离**：写权限子 agent 在独立 git worktree + 新分支工作，干净则自动清理，有改动则保留供 review
- **六层权限系统**：hard-deny → deny rules → read-only → allow rules → session → ask，规则支持 glob 匹配，persist 写入 `.coder/settings.json`
- **Hooks 系统**：`PreToolUse` / `PostToolUse` 钩子，支持 shell 命令 + 回调；内置 `dangerous_bash_guard` 和 `file_write_audit`
- **Token / 成本追踪**：每 turn 打印 `in / out / cache_read ~$0.0012`，`/tokens` 显示累计
- **LLM 摘要压缩**：监控 token 占比触发 auto-compact，历史对话压缩为结构化摘要
- **工具结果持久化**：大输出自动 offload 到 `~/.coder/sessions/`，消息里只留 preview
- **会话持久化**：JSONL 存储，`--resume` 断点续话，`--list` 查看历史
- **环境感知**：系统提示词注入 cwd、git 分支、OS、Shell、当前日期

## 代码结构

```
src/coder/
├── cli/
│   ├── repl.py          # 主循环、make_agent()、CLI 入口
│   ├── render.py        # ANSI 渲染、流事件显示、子 agent 进度
│   └── commands.py      # slash 命令（/compact /tokens /clear …）
│
├── core/
│   ├── agent_loop.py    # Agent 主循环：call_llm → exec_tools → repeat
│   ├── context.py       # 对话历史 + 系统提示词（含环境信息）
│   ├── services.py      # AgentServices：session 级依赖容器
│   ├── streaming.py     # 流式工具执行，concurrent-safe 并发调度
│   ├── events.py        # 流事件类型：TextDelta / ToolUseStart / …
│   └── errors.py        # PermissionDeniedError
│
├── tools/
│   ├── base.py          # Tool 抽象基类、ToolResult、ToolUseBlock
│   ├── bash.py          # 执行 shell 命令
│   ├── file_read.py     # 读文件
│   ├── file_edit.py     # 精确字符串替换
│   ├── file_write.py    # 写文件
│   ├── glob_tool.py     # 文件路径匹配
│   ├── grep_tool.py     # 正则内容搜索
│   └── agent.py         # 子 agent 分派
│
├── permissions/
│   ├── manager.py       # 六层权限检查
│   └── rules.py         # 规则解析与匹配（ToolName(pattern) 格式）
│
├── hooks/
│   ├── runner.py        # HookRunner：shell command + callback hooks
│   └── builtin.py       # 内置 hook：dangerous_bash_guard / file_write_audit
│
├── subagents/
│   ├── registry.py      # AgentDefinition + 注册表
│   ├── general_purpose.py
│   ├── explore.py
│   └── plan.py
│
├── compaction/
│   └── compact.py       # LLM 摘要压缩 + auto-compact + token 估算
│
├── persistence/
│   ├── session.py       # 会话 JSONL 存储 + resume
│   ├── settings.py      # 两层配置加载/合并（user + project）
│   └── tool_results.py  # 大工具结果 offload 到磁盘
│
├── git/
│   └── worktree.py      # git worktree 创建 / 清理 / 变更检测
│
├── usage.py             # Token / 成本追踪（UsageTracker）
└── logging_config.py    # 结构化日志（console + 按 session 写文件）
```

## Slash 命令

| 命令 | 说明 |
|------|------|
| `/compact` | 手动触发 LLM 摘要压缩 |
| `/tokens` | 显示 token 用量和累计成本 |
| `/sessions` | 列出本项目的历史会话 |
| `/clear` | 清空当前对话历史 |
| `/help` | 显示命令列表 |

## 配置文件

运行时自动识别两个位置的 `settings.json`，后者覆盖前者：

- `~/.coder/settings.json` — 用户全局配置
- `<项目目录>/.coder/settings.json` — 项目配置

示例（允许所有 Read 操作，禁止 rm -rf）：

```json
{
  "permissions": {
    "allow": ["Read"],
    "deny": ["Bash(rm -rf *)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "echo file changed" }]
      }
    ]
  }
}
```

## 运行测试

```bash
python -m pytest tests/ -q
```
