"""Slash-command handlers for the REPL."""

import os
from .render import DIM, GREEN, RED, YELLOW, RESET
from ..compaction.compact import (
    estimate_tokens, compact_conversation,
    COMPACT_USER_PREFIX, _calculate_keep_index, _is_tool_result,
)
from ..persistence.session import SessionManager


async def handle_slash(agent, cmd: str):
    parts = cmd.strip().split()
    name = parts[0].lower()

    if name == "/compact":
        ctx = agent.context
        tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
        print(f"  {DIM}Current tokens (est): {tokens}{RESET}")
        print(f"  {DIM}Compacting...{RESET}")
        try:
            keep_idx = _calculate_keep_index(ctx.messages)
            to_summarize = ctx.messages[:keep_idx]
            to_keep = ctx.messages[keep_idx:]
            if not to_summarize:
                to_summarize = ctx.messages
                to_keep = []
            summary = await compact_conversation(
                agent.client, agent.model, to_summarize, ctx.build_system_prompt(),
            )
            while to_keep and _is_tool_result(to_keep[0]):
                to_keep.pop(0)
            ctx.messages = [{"role": "user", "content": COMPACT_USER_PREFIX + summary}] + to_keep
            new_tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
            print(f"  {GREEN}Compacted: {tokens} -> {new_tokens} tokens{RESET}")
        except Exception as e:
            print(f"  {RED}Compact failed: {e}{RESET}")

    elif name == "/tokens":
        ctx = agent.context
        tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
        tracker = agent._services.usage if agent._services else None
        print(f"  {DIM}Messages: {len(ctx.messages)}, Context tokens (est): {tokens:,}{RESET}")
        if tracker and tracker.turn_count > 0:
            print(f"  {DIM}Turns: {tracker.turn_count}{RESET}")
            print(f"  {DIM}Total input:       {tracker.total_input:>10,}{RESET}")
            print(f"  {DIM}Total output:      {tracker.total_output:>10,}{RESET}")
            if tracker.total_cache_read:
                print(f"  {DIM}Total cache read:  {tracker.total_cache_read:>10,}{RESET}")
            if tracker.total_cache_write:
                print(f"  {DIM}Total cache write: {tracker.total_cache_write:>10,}{RESET}")
            print(f"  {DIM}Total cost:        ${tracker.total_cost_usd:.4f}{RESET}")

    elif name == "/clear":
        agent.context.messages.clear()
        print(f"  {DIM}Conversation cleared.{RESET}")

    elif name == "/sessions":
        cwd = os.getcwd()
        sessions = SessionManager.list_sessions(cwd=cwd)
        if not sessions:
            print(f"  {DIM}No sessions found.{RESET}")
        else:
            for s in sessions[:10]:
                ts = s["modified"].strftime("%m-%d %H:%M")
                print(f"  {DIM}{s['id']}  {ts}  {s['preview']}{RESET}")

    elif name == "/help":
        print(f"  {DIM}/compact   — Compress conversation via LLM summary{RESET}")
        print(f"  {DIM}/tokens    — Show estimated token count{RESET}")
        print(f"  {DIM}/sessions  — List recent sessions{RESET}")
        print(f"  {DIM}/clear     — Clear conversation history{RESET}")
        print(f"  {DIM}/help      — Show this help{RESET}")

    else:
        print(f"  {YELLOW}Unknown command: {name}. Type /help for commands.{RESET}")
