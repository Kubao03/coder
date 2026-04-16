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
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ANTHROPIC_API_KEY` | API key | `sk-...` |
| `ANTHROPIC_BASE_URL` | API 地址（默认 Anthropic，可换兼容的模型例如：DeepSeek）| `https://api.deepseek.com/anthropic` |
| `MODEL_ID` | 模型名称 | `deepseek-chat` / `claude-opus-4-6` |

## 功能

- **流式 Agent 主循环**：SSE 实时解析，tool_use 块并发执行
- **6 个内置工具**：Bash / Read / Edit / Write / Glob / Grep
- **六层权限系统**：hard-deny → settings deny → read-only → allow rules → session → ask，规则支持 `Bash(git add *)` glob 匹配，persist 写入 `.coder/settings.json` 跨会话复用
- **LLM 摘要压缩**：监控 token 占比触发 auto-compact，历史对话压缩为结构化摘要
- **会话持久化**：JSONL 存储，支持 `--resume` 断点续话
- **环境感知**：系统提示词注入 OS、Shell、git 状态等环境信息

## 架构

```
main.py               # CLI 入口，REPL 循环，slash 命令
agent_loop.py         # Agent 主循环：call_llm → exec_tools → repeat
context.py            # 对话历史 + 系统提示词构建（含环境信息）
streaming_executor.py # 流式工具执行，concurrent-safe 并发调度
permissions.py        # 六层权限检查
permission_rule.py    # 权限规则解析与匹配 (ToolName(pattern) 格式)
settings.py           # 两层配置加载/合并/更新 (user + project)
session.py            # 会话持久化（JSONL 存储 + resume）
agent_types.py        # ToolResult, ToolUseBlock 数据类型
services/
  compact.py          # LLM 摘要压缩 + auto-compact
tools/
  base.py             # Tool 抽象基类
  bash.py             # 执行 shell 命令
  file_read.py        # 读文件
  file_edit.py        # 精确字符串替换
  file_write.py       # 写文件
  glob_tool.py        # 文件匹配
  grep_tool.py        # 正则搜索
```

## 运行测试

```bash
python -m pytest tests/ -v
```
