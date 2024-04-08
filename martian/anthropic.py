"""Anthropic Interface using create_call_anthropic.

Use this module as a base to implement the Other LLM services interface.
"""

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types import ContentBlockDeltaEvent, MessageDeltaEvent

from martian.exceptions import http_400_llm_service_error
from martian.maps.anthropic_to_martian import (
	map_anthropic_to_martian,
	map_anthropic_to_martian_chunk_end,
	map_anthropic_to_martian_content_block_event,
	map_anthropic_to_martian_start_event,
)
from martian.models.martian import Message


class AsyncAnthropicHandler:
	def __init__(self, init_kwargs: dict[Any, Any]) -> None:
		self.client = AsyncAnthropic(**init_kwargs)
		self.first_message: Message | None = None

	async def create(
		self, create_kwarg: dict[Any, Any]
	) -> Message | AsyncIterator[Message]:
		try:
			if create_kwarg.get("stream", False):
				return self._create_anthropic_stream(create_kwarg)

			return map_anthropic_to_martian(
				await self.client.messages.create(**create_kwarg)
			)
		except AnthropicError as e:
			raise http_400_llm_service_error(f"AnthropicError {e}") from e

	async def _create_anthropic_stream(
		self, create_kwarg: dict[Any, Any]
	) -> AsyncIterator[Message]:
		stream = await self.client.messages.create(**create_kwarg)
		first_event = await anext(stream)
		self.first_message = map_anthropic_to_martian_start_event(first_event)
		yield self.first_message

		async for chunk in stream:
			if isinstance(chunk, ContentBlockDeltaEvent):
				yield map_anthropic_to_martian_content_block_event(
					chunk,
					self.first_message,
				)
			elif isinstance(chunk, MessageDeltaEvent):
				yield map_anthropic_to_martian_chunk_end(
					chunk,
					self.first_message,
				)


__all__ = ["AsyncAnthropic"]
