"""OpenAI Interface using create_call_openai.

Use this module as a base to implement the Other LLM services interface.
"""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from martian.exceptions import http_400_llm_service_error
from martian.maps.openai_to_martian import (
	map_openai_to_martian,
	map_openai_to_martian_chunk,
)
from martian.models.martian import Message


async def create_call_openai(
	init_kwargs: dict[Any, Any], create_kwarg: dict[Any, Any]
) -> Message | AsyncIterator[Message]:
	client = AsyncOpenAI(**init_kwargs)
	try:
		is_stream = create_kwarg.get("stream", False)
		if is_stream:
			return _create_openai_stream(client, create_kwarg)

		return map_openai_to_martian(
			await client.chat.completions.create(**create_kwarg)
		)
	except OpenAIError as e:
		raise http_400_llm_service_error(f"OpenAI {e}") from e


async def _create_openai_stream(
	client: AsyncOpenAI, create_kwarg: dict[Any, Any]
) -> AsyncIterator[Message]:
	stream = await client.chat.completions.create(**create_kwarg)
	async for chunk in stream:
		yield map_openai_to_martian_chunk(chunk)


__all__ = ["AsyncOpenAI"]
