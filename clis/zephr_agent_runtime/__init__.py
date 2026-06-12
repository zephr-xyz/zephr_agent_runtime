import os
import sys
from multiprocessing import current_process


def _rebuild_native():
    if os.environ.get("REBUILD_NATIVE_BEFORE_USE", "1") == "0":
        return

    if "stubgen.py" in sys.argv[0]:
        return

    if not current_process().name.startswith("MainProcess"):
        return

    from clis.support.native_dependencies.rebuild_tinyllm import (
        plan_prepare_host_native_deps,
        plan_rebuild,
        run_prepare_host_native_deps,
        run_rebuild,
    )

    import anyio

    build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")

    def _do_rebuild():
        anyio.run(run_prepare_host_native_deps, plan_prepare_host_native_deps(CMAKE_BUILD_TYPE=build_type))
        anyio.run(run_rebuild, plan_rebuild(CMAKE_BUILD_TYPE=build_type))

    in_async = False
    try:
        anyio.get_current_task()
        in_async = True
    except anyio.NoEventLoopError:
        pass

    if in_async:
        import threading
        error = None

        def _rebuild_in_thread():
            nonlocal error
            try:
                _do_rebuild()
            except BaseException as exc:
                error = exc

        t = threading.Thread(target=_rebuild_in_thread)
        t.start()
        t.join()
        if error is not None:
            raise error
    else:
        _do_rebuild()


_rebuild_native()

from clis.zephr_agent_runtime.nanobind.zephr_agent_runtime_nanobind import *  # noqa: F401, F403, E402

import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Literal  # noqa: E402


ToolParamSpec = tuple[str, str, str, Sequence[str], bool]
ToolSpec = tuple[str, str, Sequence[ToolParamSpec]]
ToolExecutor = Callable[[Mapping[str, str]], str]
TraceSink = Callable[[dict[str, Any]], None]
ConversationStrategy = Literal["incremental_kv", "full_replay"]
SUPPORTED_CONVERSATION_STRATEGIES: tuple[str, ...] = (
    "incremental_kv",
    "full_replay",
)
RESERVED_CONVERSATION_STRATEGIES: tuple[str, ...] = ("single_turn_with_history",)


@dataclass(frozen=True)
class ConversationToolRuntimePolicy:
    invalidates_live_text_kv: bool = False
    preferred_continuation_mode: Literal["incremental_kv", "replay_prompt"] | None = None
    max_response_tokens_for_continuation: int | None = None
    record_in_conversation_history: bool = True


@dataclass(frozen=True)
class ConversationTool:
    name: str
    description: str
    params: Sequence[ToolParamSpec]
    execute: ToolExecutor
    runtime_policy: ConversationToolRuntimePolicy = field(default_factory=ConversationToolRuntimePolicy)

    def native_spec(self) -> ToolSpec:
        return (self.name, self.description, self.params)


@dataclass(frozen=True)
class ConversationConfig:
    system_instruction: str = ""
    tools: Sequence[ConversationTool] = field(default_factory=tuple)
    conversation_strategy: ConversationStrategy = "incremental_kv"
    reserve_output_tokens: int = 0
    max_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    max_history_turns: int = 6
    trace_sink: TraceSink | None = None


@dataclass(frozen=True)
class ConversationContext:
    latitude: float = 0.0
    longitude: float = 0.0
    heading_degrees: float = 0.0
    is_navigating: bool = False
    active_waypoint_id: str = ""
    prepare_context: bool = False


@dataclass(frozen=True)
class ConversationTurnRecord:
    """SDK-owned summary of one user turn, derived from conversation events."""

    conversation_turn_index: int
    conversation_strategy: str
    user_text: str
    final_text: str
    generation_stats: Mapping[str, Any] | None
    generations: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tool_calls: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tool_responses: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    effects: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    events: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "conversation_turn",
            "conversationTurnIndex": self.conversation_turn_index,
            "conversationStrategy": self.conversation_strategy,
            "userText": self.user_text,
            "finalText": self.final_text,
            "generationStats": dict(self.generation_stats or {}),
            "generations": [dict(item) for item in self.generations],
            "toolCalls": [dict(item) for item in self.tool_calls],
            "toolResponses": [dict(item) for item in self.tool_responses],
            "effects": [dict(item) for item in self.effects],
            "events": [dict(item) for item in self.events],
            "error": self.error,
        }


class TinyLLMConversation:
    """Small host-side analogue of the ZephrAgentRuntime conversation API."""

    _TOOL_RE = re.compile(r"<\|tool_call>call:([A-Za-z0-9_.-]+)\{(.*?)\}<tool_call\|>", re.S)
    _ARG_RE = re.compile(
        r"\"?([A-Za-z0-9_.-]+)\"?\s*:(?:"
        r"<\|\"\|>(.*?)<\|\"\|>"
        r"|\"((?:\\.|[^\"\\])*)\""
        r"|([^,{}]+)"
        r")",
        re.S,
    )

    def __init__(self, inference_engine: Any, config: ConversationConfig | None = None):
        self._engine = inference_engine
        self._config = config or ConversationConfig()
        self._validate_config()
        self._turns: list[tuple[str, str, str]] = []
        self._protocol_turns: list[tuple[str, str]] = []
        self._conversation_turn_index = 0
        self._active_turn_events: list[dict[str, Any]] | None = None
        self._active_conversation_turn_index: int | None = None
        self._live_kv_cache_ready = False
        self._closed = False

    def set_conversation_strategy(self, strategy: ConversationStrategy) -> None:
        self._validate_strategy(strategy)
        self._config = ConversationConfig(
            system_instruction=self._config.system_instruction,
            tools=self._config.tools,
            conversation_strategy=strategy,
            reserve_output_tokens=self._config.reserve_output_tokens,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            max_history_turns=self._config.max_history_turns,
            trace_sink=self._config.trace_sink,
        )
        self._validate_config()
        self._live_kv_cache_ready = False

    def send_message(
        self,
        text: str,
        context: ConversationContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Conversation is closed")

        conversation_turn_index = self._conversation_turn_index
        self._conversation_turn_index += 1
        previous_active_events = self._active_turn_events
        previous_active_conversation_turn_index = self._active_conversation_turn_index
        self._active_turn_events = []
        self._active_conversation_turn_index = conversation_turn_index
        native_tools = [tool.native_spec() for tool in self._config.tools]
        tools_by_name = {tool.name: tool for tool in self._config.tools}
        next_user_text = text
        recorded_user = False
        tools_enabled = bool(native_tools)
        aggregate: dict[str, Any] | None = None

        try:
            for turn_index in range(25):
                use_live_kv_cache = (
                    turn_index == 0
                    and tools_enabled
                    and self._config.conversation_strategy == "incremental_kv"
                    and self._live_kv_cache_ready
                )
                use_protocol_replay = (
                    turn_index == 0
                    and not use_live_kv_cache
                    and self._config.conversation_strategy == "full_replay"
                )
                prompt_user_message = (
                    next_user_text
                    if use_live_kv_cache or use_protocol_replay
                    else self._user_message_with_history(next_user_text)
                )
                result_label = f"turn-{turn_index}"
                result_kind = "tool_aware" if tools_enabled else "answer_only"
                result_used_live_kv_cache = False

                if use_live_kv_cache:
                    result_label = f"turn-{turn_index}-incremental-kv"
                    result_kind = "incremental_tool_aware"
                    result_used_live_kv_cache = True
                    suffix = self._incremental_user_turn_prompt(prompt_user_message)
                    self._emit({
                        "eventType": "generation_start",
                        "turnIndex": turn_index,
                        "label": result_label,
                        "kind": result_kind,
                        "prompt": suffix,
                        "promptChars": len(suffix),
                        "usedLiveKvCache": True,
                        "conversationStrategy": self._config.conversation_strategy,
                    })
                    try:
                        result = self._engine.continue_tool_aware_text_stats(
                            native_tools,
                            suffix,
                            max_tokens=self._config.max_tokens,
                            temperature=self._config.temperature,
                            top_k=self._config.top_k,
                            top_p=self._config.top_p,
                            reserve_output_tokens=self._config.reserve_output_tokens,
                        )
                    except Exception as exc:
                        self._emit({
                            "eventType": "generation_failure",
                            "turnIndex": turn_index,
                            "label": result_label,
                            "kind": result_kind,
                            "reason": str(exc),
                            "usedLiveKvCache": True,
                            "conversationStrategy": self._config.conversation_strategy,
                        })
                        raise
                elif use_protocol_replay:
                    result_label = f"turn-{turn_index}-full-replay"
                    result_kind = "tool_aware_full_replay" if tools_enabled else "answer_only_full_replay"
                    prompt = self._protocol_prompt(prompt_user_message, include_tools=tools_enabled)
                    self._emit({
                        "eventType": "generation_start",
                        "turnIndex": turn_index,
                        "label": result_label,
                        "kind": result_kind,
                        "prompt": prompt,
                        "promptChars": len(prompt),
                        "usedLiveKvCache": False,
                        "conversationStrategy": self._config.conversation_strategy,
                    })
                    if tools_enabled:
                        result = self._engine.generate_tool_aware_text_from_prompt_stats(
                            native_tools,
                            prompt,
                            max_tokens=self._config.max_tokens,
                            temperature=self._config.temperature,
                            top_k=self._config.top_k,
                            top_p=self._config.top_p,
                            reserve_output_tokens=self._config.reserve_output_tokens,
                        )
                    else:
                        result = self._engine.generate_text_from_prompt_stats(
                            prompt,
                            max_tokens=self._config.max_tokens,
                            temperature=self._config.temperature,
                            top_k=self._config.top_k,
                            top_p=self._config.top_p,
                        )
                elif tools_enabled:
                    result = self._generate_tool_aware(native_tools, prompt_user_message)
                else:
                    result = self._engine.generate_text_stats(
                        prompt_user_message,
                        self._config.system_instruction,
                        max_tokens=self._config.max_tokens,
                        temperature=self._config.temperature,
                        top_k=self._config.top_k,
                        top_p=self._config.top_p,
                    )

                self._emit_result(turn_index, result_label, result_kind, result, result_used_live_kv_cache)
                aggregate = self._merge_stats(aggregate, result)
                response = str(result.get("response") or "").strip()
                tool_calls = self._parse_tool_calls(response) if tools_enabled else []

                if not tool_calls:
                    if not recorded_user:
                        self._turns.append(("user", "", text))
                        self._protocol_turns.append(("user", text))
                        recorded_user = True
                    self._turns.append(("assistant", "", response))
                    self._protocol_turns.append(("model", response))
                    self._live_kv_cache_ready = (
                        self._config.conversation_strategy == "incremental_kv"
                        and tools_enabled
                    )
                    final = {"contents": response, "generationStats": aggregate, "isFinal": True}
                    self._emit_turn_completed(conversation_turn_index, text, response, aggregate)
                    return final

                tool_responses: list[tuple[str, str, ConversationToolRuntimePolicy]] = []
                for name, arguments in tool_calls:
                    self._emit({
                        "eventType": "tool_call",
                        "turnIndex": turn_index,
                        "name": name,
                        "arguments": dict(arguments),
                    })
                    tool = tools_by_name.get(name)
                    if tool is None:
                        raise RuntimeError(f"Unknown tool requested by model: {name}")
                    tool_response = str(tool.execute(arguments))
                    self._emit({
                        "eventType": "tool_response",
                        "turnIndex": turn_index,
                        "name": name,
                        "response": tool_response,
                        "responseChars": len(tool_response),
                    })
                    tool_responses.append((name, tool_response, tool.runtime_policy))

                if not recorded_user:
                    self._turns.append(("user", "", text))
                    self._protocol_turns.append(("user", text))
                    recorded_user = True
                for name, tool_response, _runtime_policy in tool_responses:
                    self._turns.append(("tool", name, tool_response))

                continued = self._continue_after_tool_response(turn_index, result, tool_responses)
                if continued and str(continued.get("response") or "").strip():
                    aggregate = self._merge_stats(aggregate, continued)
                    final_response = str(continued.get("response") or "").strip()
                    self._turns.append(("assistant", "", final_response))
                    self._protocol_turns.append((
                        "model",
                        self._model_protocol_body_after_tool_response(result, tool_responses, final_response),
                    ))
                    self._live_kv_cache_ready = (
                        self._tool_response_continuation_mode(
                            self._config.conversation_strategy,
                            tool_responses,
                        )
                        == "incremental_kv"
                    )
                    final = {"contents": final_response, "generationStats": aggregate, "isFinal": True}
                    self._emit_turn_completed(conversation_turn_index, text, final_response, aggregate)
                    return final

                next_user_text = self._continuation_prompt(text, tool_responses)
                tools_enabled = False
                self._live_kv_cache_ready = False

            raise RuntimeError("Exceeded recurring tool call limit")
        except Exception as exc:
            self._emit_turn_completed(conversation_turn_index, text, "", aggregate, error=str(exc))
            raise
        finally:
            self._active_turn_events = previous_active_events
            self._active_conversation_turn_index = previous_active_conversation_turn_index

    def close(self) -> None:
        self._closed = True

    def _generate_tool_aware(self, native_tools: Sequence[ToolSpec], user_message: str) -> dict[str, Any]:
        return self._engine.generate_tool_aware_text_stats(
            native_tools,
            user_message,
            self._config.system_instruction,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            reserve_output_tokens=self._config.reserve_output_tokens,
        )

    @staticmethod
    def _conversation_context(context: ConversationContext | Mapping[str, Any]) -> ConversationContext:
        if isinstance(context, ConversationContext):
            return context
        return ConversationContext(
            latitude=float(context.get("latitude", context.get("lat", 0.0)) or 0.0),
            longitude=float(context.get("longitude", context.get("lon", 0.0)) or 0.0),
            heading_degrees=float(
                context.get("heading_degrees", context.get("headingDegrees", context.get("heading", 0.0))) or 0.0
            ),
            is_navigating=bool(context.get("is_navigating", context.get("isNavigating", False))),
            active_waypoint_id=str(
                context.get("active_waypoint_id", context.get("activeWaypointId", "")) or ""
            ),
            prepare_context=bool(context.get("prepare_context", context.get("prepareContext", False))),
        )

    def _protocol_prompt(self, current_text: str, *, include_tools: bool) -> str:
        parts = ["<bos>"]
        if self._config.system_instruction:
            parts.append("<|turn>system\n")
            parts.append(self._config.system_instruction)
            if include_tools and self._config.tools:
                parts.append("<|tool>")
                for tool in self._config.tools:
                    parts.append(self._format_tool_declaration(tool))
                parts.append("<tool|>")
            parts.append("<turn|>\n")
        elif include_tools and self._config.tools:
            parts.append("<|turn>system\n<|tool>")
            for tool in self._config.tools:
                parts.append(self._format_tool_declaration(tool))
            parts.append("<tool|><turn|>\n")

        for role, body in self._history_slice(self._protocol_turns):
            parts.append(f"<|turn>{role}\n")
            parts.append(body)
            parts.append("<turn|>\n")

        parts.append("<|turn>user\n")
        parts.append(current_text)
        parts.append("<turn|>\n<|turn>model\n")
        return "".join(parts)

    def _continue_after_tool_response(
        self,
        turn_index: int,
        first_pass: Mapping[str, Any],
        tool_responses: Sequence[tuple[str, str, ConversationToolRuntimePolicy]],
    ) -> dict[str, Any]:
        tool_response_text = "".join(self._format_tool_response(name, text) for name, text, _policy in tool_responses)
        strategy = self._config.conversation_strategy
        continuation_mode = self._tool_response_continuation_mode(strategy, tool_responses)
        try:
            if continuation_mode == "incremental_kv":
                prompt = self._incremental_tool_response_prompt(first_pass, tool_response_text)
                label = "turn-tool-response-incremental-kv"
                self._emit({
                    "eventType": "generation_start",
                    "turnIndex": turn_index,
                    "label": label,
                    "kind": "tool_response_incremental_kv",
                    "prompt": prompt,
                    "promptChars": len(prompt),
                    "usedLiveKvCache": True,
                    "conversationStrategy": strategy,
                })
                result = self._engine.continue_after_tool_response_stats(
                    tool_response_text,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_k=self._config.top_k,
                    top_p=self._config.top_p,
                    reserve_output_tokens=self._config.reserve_output_tokens,
                )
                trace_result = dict(result)
                trace_result["prompt"] = prompt
                self._emit_result(turn_index, label, "tool_response_incremental_kv", trace_result, True)
                return result

            prompt = self._replay_prompt_after_tool_response(first_pass, tool_response_text)
            label = "turn-tool-response-replay"
            self._emit({
                "eventType": "generation_start",
                "turnIndex": turn_index,
                "label": label,
                "kind": "tool_response_replay",
                "prompt": prompt,
                "promptChars": len(prompt),
                "usedLiveKvCache": False,
                "conversationStrategy": strategy,
            })
            result = self._engine.generate_text_from_prompt_stats(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_k=self._config.top_k,
                top_p=self._config.top_p,
            )
            self._emit_result(turn_index, label, "tool_response_replay", result, False)
            return result
        except Exception as exc:
            self._emit({
                "eventType": "generation_failure",
                "turnIndex": turn_index,
                "label": (
                    "turn-tool-response-incremental-kv"
                    if continuation_mode == "incremental_kv"
                    else "turn-tool-response-replay"
                ),
                "kind": "tool_response_incremental_kv" if continuation_mode == "incremental_kv" else "tool_response_replay",
                "reason": str(exc),
                "usedLiveKvCache": continuation_mode == "incremental_kv",
                "conversationStrategy": strategy,
            })
            raise

    @staticmethod
    def _tool_response_continuation_mode(
        strategy: ConversationStrategy,
        tool_responses: Sequence[tuple[str, str, ConversationToolRuntimePolicy]],
    ) -> Literal["incremental_kv", "replay_prompt"]:
        for _name, _text, policy in tool_responses:
            if policy.preferred_continuation_mode:
                return policy.preferred_continuation_mode
        if any(policy.invalidates_live_text_kv for _name, _text, policy in tool_responses):
            return "replay_prompt"
        return "incremental_kv" if strategy == "incremental_kv" else "replay_prompt"

    def _user_message_with_history(self, current_text: str) -> str:
        history = self._history_slice(self._turns)
        if not history:
            return current_text
        lines = ["", "Conversation history:"]
        for kind, name, text in history:
            if kind == "user":
                lines.append(f"User: {text}")
            elif kind == "assistant":
                lines.append(f"Assistant: {text}")
            elif kind == "tool":
                lines.append(f"Tool {name}: {text}")
        lines.extend(["", "Current user request:", current_text])
        return "\n".join(lines)

    def _history_slice(self, values: Sequence[Any]) -> Sequence[Any]:
        max_history_turns = max(0, self._config.max_history_turns)
        if max_history_turns == 0:
            return []
        user_turns = 0
        for index in range(len(values) - 1, -1, -1):
            if values[index][0] == "user":
                user_turns += 1
                if user_turns >= max_history_turns:
                    return values[index:]
        return values

    @staticmethod
    def _continuation_prompt(
        original_user_text: str,
        responses: Sequence[tuple[str, str, ConversationToolRuntimePolicy]],
    ) -> str:
        lines = ["Original user request:", original_user_text, "", "Tool results:"]
        lines.extend(f"{name}: {text}" for name, text, _policy in responses)
        lines.extend(["", "Use the tool results to answer the original request concisely."])
        return "\n".join(lines)

    @staticmethod
    def _incremental_user_turn_prompt(user_message: str) -> str:
        return f"<turn|>\n<|turn>user\n{user_message}<turn|>\n<|turn>model\n"

    @staticmethod
    def _incremental_tool_response_prompt(first_pass: Mapping[str, Any], tool_response_text: str) -> str:
        response = str(first_pass.get("response") or "")
        separator = "" if "<|tool_response>" in response else "<|tool_response>"
        return f"{separator}{tool_response_text}<tool_response|>"

    def _model_protocol_body_after_tool_response(
        self,
        first_pass: Mapping[str, Any],
        tool_responses: Sequence[tuple[str, str, ConversationToolRuntimePolicy]],
        final_response: str,
    ) -> str:
        tool_response_text = "".join(self._format_tool_response(name, text) for name, text, _policy in tool_responses)
        response = str(first_pass.get("response") or "")
        separator = "" if "<|tool_response>" in response else "<|tool_response>"
        return f"{response}{separator}{tool_response_text}<tool_response|>{final_response}"

    @staticmethod
    def _replay_prompt_after_tool_response(first_pass: Mapping[str, Any], tool_response_text: str) -> str:
        response = str(first_pass.get("response") or "")
        prompt = str(first_pass.get("prompt") or "")
        separator = "" if "<|tool_response>" in response else "<|tool_response>"
        return f"{prompt}{response}{separator}{tool_response_text}<tool_response|>"

    @staticmethod
    def _format_tool_declaration(tool: ConversationTool) -> str:
        props = []
        required = []
        for name, description, param_type, enum_values, is_required in sorted(tool.params, key=lambda p: p[0]):
            type_name = (param_type or "STRING").upper()
            prop = (
                f"{name}:{{description:<|\"|>{TinyLLMConversation._escape_tool_text(description)}<|\"|>,"
                f"type:<|\"|>{type_name}<|\"|>"
            )
            if enum_values:
                enum_text = ",".join(
                    f"<|\"|>{TinyLLMConversation._escape_tool_text(value)}<|\"|>"
                    for value in enum_values
                )
                prop += f",enum:[{enum_text}]"
            prop += "}"
            props.append(prop)
            if is_required:
                required.append(f"<|\"|>{name}<|\"|>")

        return (
            f"declaration:{tool.name}{{"
            f"description:<|\"|>{TinyLLMConversation._escape_tool_text(tool.description)}<|\"|>,"
            f"parameters:{{properties:{{{','.join(props)}}},"
            f"required:[{','.join(required)}],"
            "type:<|\"|>OBJECT<|\"|>}}"
        )

    @staticmethod
    def _format_tool_response(name: str, response: str) -> str:
        text = response.strip()
        if text.startswith("{") and text.endswith("}"):
            return f"response:{name}{text}"
        return f"response:{name}{{result:<|\"|>{TinyLLMConversation._escape_tool_text(text)}<|\"|>}}"

    @staticmethod
    def _escape_tool_text(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
            .replace("\t", "\\t")
        )

    def _parse_tool_calls(self, response: str) -> list[tuple[str, dict[str, str]]]:
        return [
            (match.group(1), self._parse_tool_args(match.group(2)))
            for match in self._TOOL_RE.finditer(response)
        ]

    def _parse_tool_args(self, body: str) -> dict[str, str]:
        return {
            match.group(1): next(
                value for value in match.groups()[1:] if value is not None
            ).strip()
            for match in self._ARG_RE.finditer(body)
        }

    def _emit_result(
        self,
        turn_index: int,
        label: str,
        kind: str,
        result: Mapping[str, Any],
        used_live_kv_cache: bool,
    ) -> None:
        prompt = str(result.get("prompt") or "")
        response = str(result.get("response") or "")
        self._emit({
            "eventType": "generation_result",
            "turnIndex": turn_index,
            "label": label,
            "kind": kind,
            "prompt": prompt,
            "promptChars": len(prompt),
            "response": response,
            "responseChars": len(response),
            "prefillTokens": int(result.get("prefill_tokens") or 0),
            "stats": self._stage_stats(result),
            "usedLiveKvCache": used_live_kv_cache,
            "conversationStrategy": self._config.conversation_strategy,
            "currentPos": getattr(self._engine, "current_pos", None),
        })
        self._emit_memory_sample(turn_index=turn_index, label=f"{label}:after_generation")

    @staticmethod
    def _stage_stats(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tokenizeMs": int(result.get("tokenize_ms") or 0),
            "prefillMs": int(result.get("prefill_ms") or 0),
            "decodeMs": int(result.get("decode_ms") or 0),
            "firstDecodeMs": int(result.get("first_decode_ms") or 0),
            "inputTokens": int(result.get("input_ids_count") or 0),
            "outputTokens": int(result.get("decode_steps") or 0),
            "constrainedTargetDecodeCalls": int(result.get("constrained_target_decode_calls") or 0),
        }

    @staticmethod
    def _merge_stats(aggregate: dict[str, Any] | None, result: Mapping[str, Any]) -> dict[str, Any]:
        current = {
            "prefillTokens": int(result.get("prefill_tokens") or 0),
            "decodeTokens": int(result.get("decode_steps") or 0),
            "prefillMs": int(result.get("prefill_ms") or 0),
            "decodeMs": int(result.get("decode_ms") or 0),
            "firstDecodeMs": int(result.get("first_decode_ms") or 0),
            "modelCalls": 1,
        }
        if aggregate is None:
            return current
        return {
            "prefillTokens": int(aggregate.get("prefillTokens") or 0) + current["prefillTokens"],
            "decodeTokens": int(aggregate.get("decodeTokens") or 0) + current["decodeTokens"],
            "prefillMs": int(aggregate.get("prefillMs") or 0) + current["prefillMs"],
            "decodeMs": int(aggregate.get("decodeMs") or 0) + current["decodeMs"],
            "firstDecodeMs": int(aggregate.get("firstDecodeMs") or 0) or current["firstDecodeMs"],
            "modelCalls": int(aggregate.get("modelCalls") or 0) + 1,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if event.get("eventType") == "generation_start":
            self._emit_memory_sample(
                turn_index=event.get("turnIndex") if isinstance(event.get("turnIndex"), int) else None,
                label=f"{event.get('label') or 'generation'}:before_generation",
            )
        event.setdefault("conversationStrategy", self._config.conversation_strategy)
        if self._active_conversation_turn_index is not None:
            event.setdefault("conversationTurnIndex", self._active_conversation_turn_index)
        if self._active_turn_events is not None and event.get("eventType") != "turn_completed":
            self._active_turn_events.append(dict(event))
        sink = self._config.trace_sink
        if sink is None:
            return
        event.setdefault("timestampMs", int(time.time() * 1000))
        sink(event)

    def _emit_memory_sample(self, *, turn_index: int | None, label: str) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss = int(usage.ru_maxrss)
        if sys.platform == "darwin":
            peak_rss_mb = peak_rss // (1024 * 1024)
        else:
            peak_rss_mb = peak_rss // 1024
        self._emit({
            "eventType": "memory_sample",
            "turnIndex": turn_index,
            "label": label,
            "memory": {
                "label": label,
                "peakRssMb": peak_rss_mb,
            },
        })

    def _emit_turn_completed(
        self,
        conversation_turn_index: int,
        user_text: str,
        final_text: str,
        generation_stats: Mapping[str, Any] | None,
        *,
        error: str | None = None,
    ) -> None:
        events = [dict(event) for event in self._active_turn_events or []]
        tool_calls = [
            {
                "turnIndex": event.get("turnIndex"),
                "name": event.get("name"),
                "arguments": dict(event.get("arguments") or {}),
            }
            for event in events
            if event.get("eventType") == "tool_call"
        ]
        tool_responses = [
            {
                "turnIndex": event.get("turnIndex"),
                "name": event.get("name"),
                "response": event.get("response", ""),
                "responseChars": event.get("responseChars", 0),
            }
            for event in events
            if event.get("eventType") == "tool_response"
        ]
        generations = [
            {
                "turnIndex": event.get("turnIndex"),
                "label": event.get("label", ""),
                "kind": event.get("kind", ""),
                "prompt": event.get("prompt", ""),
                "response": event.get("response", ""),
                "stats": dict(event.get("stats") or {}),
                "usedLiveKvCache": bool(event.get("usedLiveKvCache", False)),
                "conversationStrategy": event.get("conversationStrategy", self._config.conversation_strategy),
            }
            for event in events
            if event.get("eventType") == "generation_result"
        ]
        effects = [
            {key: value for key, value in event.items() if key != "eventType"}
            for event in events
            if event.get("eventType") == "effect_applied"
        ]
        record = ConversationTurnRecord(
            conversation_turn_index=conversation_turn_index,
            conversation_strategy=self._config.conversation_strategy,
            user_text=user_text,
            final_text=final_text,
            generation_stats=generation_stats,
            generations=generations,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            effects=effects,
            events=events,
            error=error,
        ).to_dict()
        self._emit({
            "eventType": "turn_completed",
            "conversationTurnIndex": conversation_turn_index,
            "turn": record,
            "record": record,
            "error": error,
        })

    def _validate_config(self) -> None:
        self._validate_strategy(self._config.conversation_strategy)

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        if strategy in RESERVED_CONVERSATION_STRATEGIES:
            raise ValueError(f"{strategy} is reserved and is not implemented yet")
        if strategy not in SUPPORTED_CONVERSATION_STRATEGIES:
            supported = ", ".join(SUPPORTED_CONVERSATION_STRATEGIES)
            raise ValueError(f"unknown conversation strategy {strategy!r}; expected one of: {supported}")

def create_tinyllm_conversation(
    inference_engine: Any,
    config: ConversationConfig | None = None,
) -> TinyLLMConversation:
    return TinyLLMConversation(inference_engine, config)


class PyTorchConversationEngine:
    """Hugging Face/PyTorch engine adapter for the Python conversation API.

    The adapter deliberately exposes the same small method surface as the
    host TinyLLM binding. Incremental continuation is currently emulated by
    materializing the full prompt text before each generation; traces still
    preserve which suffix the conversation API would have sent to a live KV
    backend.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        device: Any = None,
        max_new_tokens: int = 192,
        generate_fn: Callable[[str, int, float, int, float], tuple[str, int, int, str]] | None = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generate_fn = generate_fn
        self.current_pos = 0
        self._cache_text = ""
        self._last_response = ""

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        max_new_tokens: int = 192,
        dtype: Any = None,
        device_map: str = "auto",
    ) -> "PyTorchConversationEngine":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs: dict[str, Any] = {"device_map": device_map}
        model_kwargs["dtype"] = dtype if dtype is not None else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=getattr(model, "device", None) or torch.device("cpu"),
            max_new_tokens=max_new_tokens,
        )

    def reset(self) -> None:
        self.current_pos = 0
        self._cache_text = ""
        self._last_response = ""

    def generate_tool_aware_text_stats(
        self,
        native_tools: Sequence[ToolSpec],
        user_message: str,
        system_instruction: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        reserve_output_tokens: int = 0,
    ) -> dict[str, Any]:
        del reserve_output_tokens
        prompt = self._protocol_prompt(system_instruction, native_tools, user_message)
        return self.generate_text_from_prompt_stats(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def generate_tool_aware_text_from_prompt_stats(
        self,
        native_tools: Sequence[ToolSpec],
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        reserve_output_tokens: int = 0,
    ) -> dict[str, Any]:
        del native_tools, reserve_output_tokens
        return self.generate_text_from_prompt_stats(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def generate_text_stats(
        self,
        user_message: str,
        system_instruction: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        prompt = self._protocol_prompt(system_instruction, (), user_message)
        return self.generate_text_from_prompt_stats(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def generate_text_from_prompt_stats(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        started = time.monotonic()
        response, input_tokens, output_tokens, stop_reason = self._generate(
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        )
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        self._cache_text = prompt + response
        self._last_response = response
        self.current_pos = input_tokens + output_tokens
        return {
            "prompt": prompt,
            "response": response,
            "prefill_tokens": input_tokens,
            "decode_steps": output_tokens,
            "input_ids_count": input_tokens,
            "prefill_ms": elapsed_ms,
            "decode_ms": 0,
            "first_decode_ms": 0,
            "stop_reason": stop_reason,
        }

    def continue_tool_aware_text_stats(
        self,
        native_tools: Sequence[ToolSpec],
        prompt_suffix: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        reserve_output_tokens: int = 0,
    ) -> dict[str, Any]:
        del native_tools, reserve_output_tokens
        return self._continue_from_suffix(
            prompt_suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def continue_after_tool_response_stats(
        self,
        tool_response_text: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        reserve_output_tokens: int = 0,
    ) -> dict[str, Any]:
        del reserve_output_tokens
        separator = "" if self._cache_text.endswith("<|tool_response>") else "<|tool_response>"
        suffix = f"{separator}{tool_response_text}<tool_response|>"
        return self._continue_from_suffix(
            suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def _continue_from_suffix(
        self,
        suffix: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> dict[str, Any]:
        full_prompt = self._cache_text + suffix
        result = self.generate_text_from_prompt_stats(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        result["prompt"] = suffix
        return result

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[str, int, int, str]:
        if self.generate_fn is not None:
            return self.generate_fn(prompt, max_tokens, temperature, top_k, top_p)

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to") and self.device is not None:
            inputs = inputs.to(self.device)
        eos = self.tokenizer.eos_token_id
        eos_ids = [eos, 106] if eos is not None else [106]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=self.tokenizer.pad_token_id or eos,
            )
        input_tokens = int(inputs.input_ids.shape[-1])
        generated = output_ids[0][input_tokens:]
        output_tokens = int(generated.shape[-1])
        stop_reason = "empty"
        if output_tokens:
            last_token = int(generated[-1].item())
            stop_reason = "eos" if last_token in eos_ids else "max_new_tokens"
            if output_tokens < min(max_tokens, self.max_new_tokens) and stop_reason != "eos":
                stop_reason = "stopped"
        return (
            self.tokenizer.decode(generated, skip_special_tokens=False).strip(),
            input_tokens,
            output_tokens,
            stop_reason,
        )

    @staticmethod
    def _protocol_prompt(
        system_instruction: str,
        native_tools: Sequence[ToolSpec],
        user_message: str,
    ) -> str:
        parts = ["<bos>"]
        if system_instruction or native_tools:
            parts.append("<|turn>system\n")
            if system_instruction:
                parts.append(system_instruction)
            if native_tools:
                parts.append("<|tool>")
                for name, description, params in native_tools:
                    tool = ConversationTool(name=name, description=description, params=params, execute=lambda _args: "")
                    parts.append(TinyLLMConversation._format_tool_declaration(tool))
                parts.append("<tool|>")
            parts.append("<turn|>\n")
        parts.append("<|turn>user\n")
        parts.append(user_message)
        parts.append("<turn|>\n<|turn>model\n")
        return "".join(parts)


def create_pytorch_conversation(
    engine: PyTorchConversationEngine,
    config: ConversationConfig | None = None,
) -> TinyLLMConversation:
    return TinyLLMConversation(engine, config)
