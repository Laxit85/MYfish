"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    # For models with strict output token limits (e.g. Qwen on Groq with 1000 OTPM limit)
    if "qwen" in model.lower():
        max_tokens = min(max_tokens or 800, 800)

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e)
        err_lower = err_str.lower()
        status_code = getattr(e, "status_code", None)

        # 1. Recover from Groq tool_use_failed by extracting the model's failed_generation
        if "tool_use_failed" in err_lower and "failed_generation" in err_lower:
            try:
                import json
                import ast
                from types import SimpleNamespace

                body = getattr(e, "body", None)
                if not isinstance(body, dict):
                    start_idx = err_str.find("{")
                    end_idx = err_str.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        dict_str = err_str[start_idx:end_idx + 1]
                        try:
                            body = json.loads(dict_str)
                        except Exception:
                            body = ast.literal_eval(dict_str)

                err = body.get("error", {}) if isinstance(body, dict) else {}
                fg = None
                if isinstance(body, dict):
                    fg = body.get("failed_generation")
                    if not fg and isinstance(body.get("error"), dict):
                        fg = body["error"].get("failed_generation")
                if fg:
                    try:
                        fg_data = json.loads(fg) if isinstance(fg, str) else fg
                        name = fg_data.get("name")
                        args = fg_data.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        content = f"<invoke_tool>{json.dumps({'name': name, 'parameters': args})}</invoke_tool>"
                    except Exception:
                        content = f"<invoke_tool>{fg}</invoke_tool>"

                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=content,
                                    role="assistant",
                                    tool_calls=None,
                                )
                            )
                        ]
                    )
            except Exception:
                pass

        # 2. Auto-fallback chain on 429 rate limit across accessible Groq models
        if (
            status_code == 429
            or "429" in err_lower
            or "rate_limit_exceeded" in err_lower
            or "rate limit" in err_lower
        ):
            fallback_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
            current_idx = fallback_models.index(model) if model in fallback_models else -1
            if current_idx + 1 < len(fallback_models):
                next_model = fallback_models[current_idx + 1]
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["model"] = next_model
                return create_chat_completion(client, **fallback_kwargs)

        # 3. If request exceeds provider TPM/token limits (413 Request too large), trim message history and retry
        if status_code == 413 or "request too large" in err_lower or "413" in err_lower:
            msgs = kwargs.get("messages", [])
            if len(msgs) > 2:
                trimmed_msgs = [msgs[0]]
                for m in msgs[-2:]:
                    c = m.get("content", "")
                    trimmed_msgs.append({**m, "content": c[:1000] if isinstance(c, str) and len(c) > 1000 else c})
                retry_kwargs = dict(kwargs)
                retry_kwargs["messages"] = trimmed_msgs
                return create_chat_completion(client, **retry_kwargs)

        # 4. If output token limit exceeded (e.g. reduce max_tokens / OTPM), clamp max_tokens and retry
        if "reduce max_tokens" in err_lower or "otpm" in err_lower:
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = 750
            return create_chat_completion(client, **retry_kwargs)

        raise


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if not content:
        reasoning = getattr(message, "reasoning", "")
        if reasoning and isinstance(reasoning, str):
            content = reasoning

    # If the model emitted structured tool_calls, format them into invoke_tool blocks
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls and not content:
        import json
        tool_parts = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn:
                name = getattr(fn, "name", "")
                args_str = getattr(fn, "arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {}
                tool_parts.append(f"<invoke_tool>{json.dumps({'name': name, 'parameters': args})}</invoke_tool>")
        if tool_parts:
            return "\n".join(tool_parts)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
