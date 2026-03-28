from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import httpx
from rich import print

from xiaogpt.bot.base_bot import BaseBot, ChatHistoryMixin
from xiaogpt.utils import split_sentences

if TYPE_CHECKING:
    import openai


MINIMAX_API_BASE = "https://api.minimax.chat/v1"


@dataclasses.dataclass
class MiniMaxBot(ChatHistoryMixin, BaseBot):
    name: ClassVar[str] = "MiniMax"
    default_options: ClassVar[dict[str, str]] = {"model": "MiniMax-Text-01"}
    minimax_api_key: str
    minimax_model: str = "MiniMax-Text-01"
    proxy: str | None = None
    history: list[tuple[str, str]] = dataclasses.field(default_factory=list, init=False)

    def _make_openai_client(self, sess: httpx.AsyncClient) -> openai.AsyncOpenAI:
        import openai

        return openai.AsyncOpenAI(
            api_key=self.minimax_api_key,
            http_client=sess,
            base_url=MINIMAX_API_BASE,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            minimax_api_key=config.minimax_api_key,
            minimax_model=config.minimax_model,
            proxy=config.proxy,
        )

    async def ask(self, query, **options):
        ms = self.get_messages()
        ms.append({"role": "user", "content": f"{query}"})
        kwargs = {**self.default_options, **options}
        kwargs["model"] = self.minimax_model
        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy
        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            client = self._make_openai_client(sess)
            try:
                completion = await client.chat.completions.create(messages=ms, **kwargs)
            except Exception as e:
                print(str(e))
                return ""

            message = completion.choices[0].message.content
            self.add_message(query, message)
            print(message)
            return message

    async def ask_stream(self, query, **options):
        ms = self.get_messages()
        ms.append({"role": "user", "content": f"{query}"})
        kwargs = {**self.default_options, **options}
        kwargs["model"] = self.minimax_model
        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy
        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            client = self._make_openai_client(sess)
            try:
                completion = await client.chat.completions.create(
                    messages=ms, stream=True, **kwargs
                )
            except Exception as e:
                print(str(e))
                return

            async def text_gen():
                async for event in completion:
                    if not event.choices:
                        continue
                    chunk_message = event.choices[0].delta
                    if chunk_message.content is None:
                        continue
                    print(chunk_message.content, end="")
                    yield chunk_message.content

            message = ""
            try:
                async for sentence in split_sentences(text_gen()):
                    message += sentence
                    yield sentence
            finally:
                print()
                self.add_message(query, message)