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
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ANTHROPIC_API_KEY` | API key | `sk-...` |
| `ANTHROPIC_BASE_URL` | API 地址（默认 Anthropic，可换 DeepSeek）| `https://api.deepseek.com/anthropic` |
| `MODEL_ID` | 模型名称 | `deepseek-chat` / `claude-opus-4-5` |

## 架构

```
main.py          # CLI 入口，REPL 循环
agent_loop.py    # Agent 主循环：call_llm → exec_tools → repeat
context.py       # 对话历史 + 系统提示词构建
permissions.py   # 五层权限检查
agent_types.py   # ToolResult, ToolUseBlock 数据类型
tools/
  base.py        # Tool 抽象基类
  bash.py        # 执行 shell 命令
  file_read.py   # 读文件
  file_edit.py   # 精确字符串替换
  file_write.py  # 写文件
  glob_tool.py   # 文件匹配
  grep_tool.py   # 正则搜索
```

## 运行测试

```bash
python -m pytest tests/ -v
```
