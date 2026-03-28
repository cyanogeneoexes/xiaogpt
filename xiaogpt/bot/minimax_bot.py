from __future__ import annotations

import dataclasses
from typing import Any, AsyncGenerator, ClassVar

import httpx
from rich import print

from xiaogpt.bot.base_bot import BaseBot, ChatHistoryMixin
from xiaogpt.utils import split_sentences


MINIMAX_API_BASE = "https://api.minimaxi.com/anthropic"


@dataclasses.dataclass
class MinimaxBot(ChatHistoryMixin, BaseBot):
    name: ClassVar[str] = "Minimax"
    default_options: ClassVar[dict[str, Any]] = {
        "model": "MiniMax-M2.7",
        "max_tokens": 2048,
        "temperature": 1.0,
    }
    minimax_api_key: str = ""
    api_base: str = MINIMAX_API_BASE
    proxy: str | None = None
    history: list[tuple[str, str]] = dataclasses.field(default_factory=list, init=False)

    def _make_anthropic_client(self) -> Any:
        import anthropic

        return anthropic.AsyncAnthropic(
            api_key=self.minimax_api_key,
            base_url=self.api_base,
        )

    @classmethod
    def from_config(cls, config):
        # Support both MINIMAX_API_KEY and ANTHROPIC_API_KEY
        api_key = config.minimax_api_key or config.anthropic_key
        # Support custom api_base via config or ANTHROPIC_BASE_URL env var
        api_base = getattr(config, 'anthropic_base_url', None) or getattr(config, 'api_base', None) or MINIMAX_API_BASE
        return cls(
            minimax_api_key=api_key,
            api_base=api_base,
            proxy=config.proxy,
        )

    def get_messages(self) -> list[dict]:
        """Get messages in Anthropic content block format."""
        ms = []
        for h in self.history:
            ms.append({"role": "user", "content": [{"type": "text", "text": h[0]}]})
            ms.append({"role": "assistant", "content": [{"type": "text", "text": h[1]}]})
        return ms

    def add_message(self, query: str, message: str) -> None:
        self.history.append([f"{query}", message])
        first_history = self.history.pop(0)
        self.history = [first_history] + self.history[-5:]

    async def ask(self, query: str, **options: Any) -> str:
        messages = self.get_messages()
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        kwargs = {**self.default_options, **options}
        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy
        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            client = self._make_anthropic_client()
            try:
                message = await client.messages.create(
                    messages=messages,
                    **kwargs,
                )
            except Exception as e:
                print(str(e))
                return ""

            # Extract text, handling ThinkingBlock
            response = ""
            for block in message.content:
                if hasattr(block, 'text') and block.text:
                    response = block.text
                    break
            self.add_message(query, response)
            print(response)
            return response

    async def ask_stream(self, query: str, **options: Any) -> AsyncGenerator[str, None]:
        messages = self.get_messages()
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        kwargs = {**self.default_options, **options}
        kwargs.pop("stream", None)

        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy

        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            client = self._make_anthropic_client()
            try:
                async with client.messages.stream(
                    messages=messages,
                    **kwargs,
                ) as stream:
                    full_text = ""
                    async for text in stream.text_stream:
                        print(text, end="")
                        full_text += text
                        yield text
                print()
                self.add_message(query, full_text)
            except Exception as e:
                print(str(e))