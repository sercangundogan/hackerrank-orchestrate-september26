"""Single model-access layer for text and image structured extraction."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai.schemas import IMAGE_AMOUNT_PROMPT_VERSION, MESSAGE_EXTRACTION_PROMPT_VERSION
from usage.tracker import UsageTracker


class ModelError(RuntimeError):
    pass


class ModelUnavailableError(ModelError):
    pass


class InvalidModelOutputError(ModelError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    payload: dict[str, Any]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    prompt_version: str


def configured_provider() -> str:
    return os.environ.get("BUYORWAIT_AI_PROVIDER", "openai").strip().lower()


def configured_text_model() -> str:
    return os.environ.get("BUYORWAIT_TEXT_MODEL", "gpt-4o-mini").strip()


def configured_vision_model() -> str:
    return os.environ.get("BUYORWAIT_VISION_MODEL", "gpt-4o-mini").strip()


def _api_key() -> str | None:
    provider = configured_provider()
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("BUYORWAIT_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    return os.environ.get("BUYORWAIT_API_KEY") or os.environ.get("OPENAI_API_KEY")


class ModelClient:
    def __init__(
        self,
        *,
        tracker: UsageTracker | None = None,
        api_key: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.tracker = tracker or UsageTracker()
        self.api_key = api_key if api_key is not None else _api_key()
        self.max_retries = max_retries
        self.provider = configured_provider()
        self.text_model = configured_text_model()
        self.vision_model = configured_vision_model()

    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        purpose: str,
        prompt_version: str,
        request_id: str | None,
        user_id: str | None,
        source_id: str | None,
        image_media_type: str | None = None,
        image_b64: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise ModelUnavailableError("no API key configured")
        model = self.vision_model if image_b64 else self.text_model
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload, usage = self._post_chat(
                    model=model,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    image_media_type=image_media_type,
                    image_b64=image_b64,
                    schema=schema,
                )
                self.tracker.record(
                    provider=self.provider,
                    model=model,
                    purpose=purpose,
                    request_id=request_id,
                    user_id=user_id,
                    source_id=source_id,
                    prompt_version=prompt_version,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    cache_hit=False,
                    success=True,
                )
                return ModelResponse(
                    payload=payload,
                    provider=self.provider,
                    model=model,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    prompt_version=prompt_version,
                )
            except (InvalidModelOutputError, ModelError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.4 * (attempt + 1))
        self.tracker.record(
            provider=self.provider,
            model=model,
            purpose=purpose,
            request_id=request_id,
            user_id=user_id,
            source_id=source_id,
            prompt_version=prompt_version,
            input_tokens=0,
            output_tokens=0,
            cache_hit=False,
            success=False,
            error=str(last_error) if last_error else "model call failed",
        )
        raise ModelError(str(last_error) if last_error else "model call failed")

    def _post_chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_media_type: str | None,
        image_b64: str | None,
        schema: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], tuple[int, int]]:
        if self.provider != "openai":
            raise ModelUnavailableError(f"provider {self.provider} is not implemented")
        user_content: Any
        if image_b64:
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type or 'image/png'};base64,{image_b64}"
                    },
                },
            ]
        else:
            user_content = user_text
        body: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ModelError(f"http {exc.code}") from exc
        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        content = raw["choices"][0]["message"]["content"]
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutputError("model did not return JSON") from exc
        if not isinstance(payload, dict):
            raise InvalidModelOutputError("model JSON must be an object")
        if schema and "required" in schema:
            missing = [key for key in schema["required"] if key not in payload]
            if missing:
                raise InvalidModelOutputError(f"missing keys {missing}")
        return payload, (input_tokens, output_tokens)


def default_prompt_versions() -> tuple[str, str]:
    return MESSAGE_EXTRACTION_PROMPT_VERSION, IMAGE_AMOUNT_PROMPT_VERSION
