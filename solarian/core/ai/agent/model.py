"""Planner model clients behind one interface.

There is no single wire standard worth standardising on. OpenAI-compatible
chat completions is the closest, and it is what most servers speak, but
Anthropic's compatibility layer drops prompt caching and extended thinking —
the two things a planner that re-reads a growing transcript of images most
needs. So the transcript is kept in a neutral form here and each client
converts it to its provider's native one.

The neutral form is deliberately small: a user turn carries text, images and
the results of the previous turn's tool calls; an assistant turn carries text,
tool calls, and the provider's own content blocks so they can be replayed
verbatim (Anthropic requires its signed thinking blocks back unchanged).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union

from solarian.core.configs import ModelEndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class ImagePart:
    jpeg: bytes
    #: What the model is told this image is, e.g. ``o12 right_wrist_camera/image``.
    label: str

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii")


Part = Union[str, ImagePart]


@dataclass
class ToolDef:
    name: str
    description: str
    #: JSON Schema of the arguments, an ``object``.
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    #: Set when the model's arguments were not valid JSON; the harness answers
    #: with an error instead of running the call.
    parse_error: str | None = None


@dataclass
class ToolResult:
    call_id: str
    content: list[Part]
    is_error: bool = False


@dataclass
class UserTurn:
    content: list[Part] = field(default_factory=list)
    #: Answers to every tool call of the preceding assistant turn.
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: The provider's own content, replayed as-is when this turn is resent.
    native: Any = None
    #: Which client produced ``native``; another client rebuilds from the fields.
    provider: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""


Message = Union[UserTurn, AssistantTurn]


class ModelClient(ABC):
    """One planner model. ``complete`` is a blocking call; the harness owns the thread."""

    provider: str = ""

    def __init__(self, cfg: ModelEndpointConfig):
        self.cfg = cfg
        self.model = cfg.model

    def api_key(self) -> str | None:
        return os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else None

    @abstractmethod
    def complete(
        self, system: str, messages: list[Message], tools: list[ToolDef]
    ) -> AssistantTurn: ...


class AnthropicClient(ModelClient):
    """Anthropic Messages API, with prompt caching and optional extended thinking.

    Three cache breakpoints: the tool list, the system prompt, and the end of
    the newest message, so each call reads everything up to the previous
    call's end from cache and writes only the new turn.
    """

    provider = "anthropic"

    def __init__(self, cfg: ModelEndpointConfig):
        super().__init__(cfg)
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=self.api_key(), base_url=cfg.base_url, timeout=cfg.timeout_s
        )

    @staticmethod
    def _parts(parts: list[Part]) -> list[dict]:
        blocks: list[dict] = []
        for part in parts:
            if isinstance(part, str):
                if part:
                    blocks.append({"type": "text", "text": part})
            else:
                blocks.append({"type": "text", "text": f"Image {part.label}:"})
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(part.jpeg).decode("ascii"),
                        },
                    }
                )
        return blocks

    def _messages(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for message in messages:
            if isinstance(message, UserTurn):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": self._parts(result.content) or [{"type": "text", "text": "ok"}],
                        "is_error": result.is_error,
                    }
                    for result in message.tool_results
                ]
                content += self._parts(message.content)
                out.append({"role": "user", "content": content})
            elif message.provider == self.provider and message.native is not None:
                out.append({"role": "assistant", "content": message.native})
            else:
                content = [{"type": "text", "text": message.text}] if message.text else []
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in message.tool_calls
                ]
                out.append({"role": "assistant", "content": content})
        if out and out[-1]["content"]:
            last = dict(out[-1]["content"][-1])
            last["cache_control"] = {"type": "ephemeral"}
            out[-1] = {**out[-1], "content": [*out[-1]["content"][:-1], last]}
        return out

    def complete(self, system, messages, tools):
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        if tool_defs:
            tool_defs[-1] = {**tool_defs[-1], "cache_control": {"type": "ephemeral"}}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": self._messages(messages),
            "tools": tool_defs,
        }
        if self.cfg.thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(self.cfg.thinking_budget)}
            kwargs["max_tokens"] = max(self.cfg.max_tokens, int(self.cfg.thinking_budget) + 1024)
        response = self._client.messages.create(**kwargs)

        text, calls = [], []
        for block in response.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        usage = response.usage
        return AssistantTurn(
            text="\n".join(text).strip(),
            tool_calls=calls,
            native=[block.model_dump(exclude_none=True) for block in response.content],
            provider=self.provider,
            usage=Usage(
                input_tokens=int(usage.input_tokens or 0)
                + int(usage.cache_read_input_tokens or 0)
                + int(usage.cache_creation_input_tokens or 0),
                output_tokens=int(usage.output_tokens or 0),
                cache_read_tokens=int(usage.cache_read_input_tokens or 0),
                cache_write_tokens=int(usage.cache_creation_input_tokens or 0),
            ),
            stop_reason=str(response.stop_reason or ""),
        )


class OpenAIChatClient(ModelClient):
    """OpenAI chat completions, which most model servers also speak.

    A ``tool`` message carries text only, so the images a tool returned travel
    in one user message right after the tool messages.
    """

    provider = "openai"

    def __init__(self, cfg: ModelEndpointConfig):
        super().__init__(cfg)
        import openai

        self._client = openai.OpenAI(
            api_key=self.api_key(), base_url=cfg.base_url, timeout=cfg.timeout_s
        )

    @staticmethod
    def _user_parts(parts: list[Part]) -> list[dict]:
        out: list[dict] = []
        for part in parts:
            if isinstance(part, str):
                if part:
                    out.append({"type": "text", "text": part})
            else:
                out.append({"type": "text", "text": f"Image {part.label}:"})
                out.append({"type": "image_url", "image_url": {"url": part.data_url()}})
        return out

    def _messages(self, system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for message in messages:
            if isinstance(message, UserTurn):
                images: list[Part] = []
                for result in message.tool_results:
                    text = "\n".join(p for p in result.content if isinstance(p, str))
                    shown = [p for p in result.content if isinstance(p, ImagePart)]
                    if shown:
                        text += "\n" + ", ".join(f"[image {p.label} follows]" for p in shown)
                        images.extend(shown)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": ("ERROR: " if result.is_error else "") + (text or "ok"),
                        }
                    )
                parts = self._user_parts(images + list(message.content))
                if parts:
                    out.append({"role": "user", "content": parts})
            else:
                entry: dict[str, Any] = {"role": "assistant", "content": message.text or None}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in message.tool_calls
                    ]
                out.append(entry)
        return out

    def complete(self, system, messages, tools):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "max_completion_tokens": self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        if self.cfg.reasoning_effort:
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            arguments, error = _parse_arguments(call.function.arguments)
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments, parse_error=error))
        usage = response.usage
        cached = 0
        if usage is not None and usage.prompt_tokens_details is not None:
            cached = int(usage.prompt_tokens_details.cached_tokens or 0)
        return AssistantTurn(
            text=(choice.message.content or "").strip(),
            tool_calls=calls,
            provider=self.provider,
            usage=Usage(
                input_tokens=int(usage.prompt_tokens or 0) if usage else 0,
                output_tokens=int(usage.completion_tokens or 0) if usage else 0,
                cache_read_tokens=cached,
            ),
            stop_reason=str(choice.finish_reason or ""),
        )


def _parse_arguments(raw: str | None) -> tuple[dict, str | None]:
    """A tool call's JSON arguments, and why they were refused if they were."""
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"arguments are not valid JSON: {exc}"
    if not isinstance(arguments, dict):
        return {}, "arguments must be a JSON object"
    return arguments, None


class OpenAIResponsesClient(ModelClient):
    """OpenAI Responses API, the one OpenAI reasoning models take tools with.

    Stateless (``store: false``): each call resends the transcript, and the
    reasoning items come back encrypted so they can be replayed with it —
    the model keeps its chain of thought across tool calls without OpenAI
    keeping the run.
    """

    provider = "openai_responses"

    def __init__(self, cfg: ModelEndpointConfig):
        super().__init__(cfg)
        import openai

        self._client = openai.OpenAI(
            api_key=self.api_key(), base_url=cfg.base_url, timeout=cfg.timeout_s
        )

    @staticmethod
    def _user_parts(parts: list[Part]) -> list[dict]:
        out: list[dict] = []
        for part in parts:
            if isinstance(part, str):
                if part:
                    out.append({"type": "input_text", "text": part})
            else:
                out.append({"type": "input_text", "text": f"Image {part.label}:"})
                out.append({"type": "input_image", "image_url": part.data_url(), "detail": "high"})
        return out

    def _input(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for message in messages:
            if isinstance(message, UserTurn):
                images: list[Part] = []
                for result in message.tool_results:
                    text = "\n".join(p for p in result.content if isinstance(p, str))
                    shown = [p for p in result.content if isinstance(p, ImagePart)]
                    if shown:
                        text += "\n" + ", ".join(f"[image {p.label} follows]" for p in shown)
                        images.extend(shown)
                    out.append(
                        {
                            "type": "function_call_output",
                            "call_id": result.call_id,
                            "output": ("ERROR: " if result.is_error else "") + (text or "ok"),
                        }
                    )
                parts = self._user_parts(images + list(message.content))
                if parts:
                    out.append({"role": "user", "content": parts})
            elif message.provider == self.provider and message.native is not None:
                out.extend(message.native)
            else:
                if message.text:
                    out.append({"role": "assistant", "content": message.text})
                out.extend(
                    {"type": "function_call", "call_id": c.id, "name": c.name, "arguments": json.dumps(c.arguments)}
                    for c in message.tool_calls
                )
        return out

    def complete(self, system, messages, tools):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self._input(messages),
            "max_output_tokens": self.cfg.max_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "tools": [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                }
                for t in tools
            ],
        }
        if self.cfg.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}
        response = self._client.responses.create(**kwargs)

        text, calls = [], []
        for item in response.output:
            if item.type == "message":
                text += [c.text for c in item.content if c.type == "output_text"]
            elif item.type == "function_call":
                arguments, error = _parse_arguments(item.arguments)
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments, parse_error=error))
        usage = response.usage
        cached = 0
        if usage is not None and usage.input_tokens_details is not None:
            cached = int(usage.input_tokens_details.cached_tokens or 0)
        incomplete = response.incomplete_details
        return AssistantTurn(
            text="\n".join(text).strip(),
            tool_calls=calls,
            native=[item.model_dump(exclude_none=True) for item in response.output],
            provider=self.provider,
            usage=Usage(
                input_tokens=int(usage.input_tokens or 0) if usage else 0,
                output_tokens=int(usage.output_tokens or 0) if usage else 0,
                cache_read_tokens=cached,
            ),
            stop_reason=str(incomplete.reason if incomplete else response.status or ""),
        )


_CLIENTS: dict[str, type[ModelClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIChatClient,
    "openai_responses": OpenAIResponsesClient,
}


def make_model_client(cfg: ModelEndpointConfig) -> ModelClient:
    try:
        cls = _CLIENTS[cfg.provider]
    except KeyError:
        raise ValueError(
            f"Unknown planner provider {cfg.provider!r}; expected one of {sorted(_CLIENTS)}"
        ) from None
    return cls(cfg)
