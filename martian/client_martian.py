from collections.abc import AsyncIterator
from typing import Any

from martian.anthropic import AsyncAnthropicHandler
from martian.enums import LLMServiceEnum
from martian.exceptions import http_400_invalid_llm_service
from martian.models.martian import Message
from martian.openai import create_call_openai


class AsyncMartian:
	def __init__(self, llm_service: LLMServiceEnum, **kwargs: Any) -> None:
		"""kwargs are the parameters needed for each specific LLM service."""
		self.llm_service = llm_service
		self.init_kwargs = kwargs

	async def create(self, **kwargs: Any) -> Message | AsyncIterator[Message]:
		"""Change the function name if you will.

		Add new llm service here creating a create_call_* function.
		If we start having a lot of services, we can change this to
			a factory pattern: dict[llm_service, Callable].
		"""
		match self.llm_service:
			case LLMServiceEnum.ANTHROPIC:
				return await AsyncAnthropicHandler(self.init_kwargs).create(
					kwargs
				)
			# OpenAI and Together uses the same lib.
			case LLMServiceEnum.OPENAI | LLMServiceEnum.TOGETHER:
				return await create_call_openai(self.init_kwargs, kwargs)
			# Add new llm service here.
			case _:
				# Add http exception here
				raise http_400_invalid_llm_service(self.llm_service)
