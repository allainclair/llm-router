from typing import Any
from unittest.mock import patch

from anthropic import AnthropicError
from anthropic.types import ContentBlock
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from fastapi import status
from httpx import AsyncClient
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class AsyncAnthropicMockBase:
	"""Using mock approach,

	I could not use VCR from pytest-recording package or respx
	"""

	def __init__(self, **kwargs: Any) -> None:
		pass

	@property
	def messages(self: "AsyncAnthropicMockBase") -> "AsyncAnthropicMockBase":
		return self

	async def create(self, **kwargs: Any) -> AnthropicMessage:  # type: ignore[empty-body]  # Missing return statement
		pass


class AsyncAnthropicMockError(AsyncAnthropicMockBase):
	async def create(self, **kwargs: Any) -> AnthropicMessage:  # noqa: ARG002
		raise AnthropicError("Mocked AnthropicError")


class AsyncAnthropicMockResponse(AsyncAnthropicMockBase):
	async def create(self, **kwargs: Any) -> AnthropicMessage:  # noqa: ARG002
		return AnthropicMessage(
			id="msg_02WuDdq0aMLimFapk5jXWnUt",
			content=[
				ContentBlock(
					text=(
						"Hi! It's nice to meet you. "
						"How can I assist you today?"
					),
					type="text",
				)
			],
			model="claude-3-opus-20240229",
			role="assistant",
			stop_reason="end_turn",
			stop_sequence=None,
			type="message",
			usage=AnthropicUsage(input_tokens=10, output_tokens=19),
		)


class AsyncOpenAIMockBase:
	"""Using mock approach,

	I could not use VCR from pytest-recording package or respx
	"""

	def __init__(self, **kwargs: Any) -> None:
		pass

	@property
	def chat(self: "AsyncOpenAIMockBase") -> "AsyncOpenAIMockBase":
		return self

	@property
	def completions(self: "AsyncOpenAIMockBase") -> "AsyncOpenAIMockBase":
		return self

	async def create(self, **kwargs: Any) -> ChatCompletion:  # type: ignore[empty-body]  # Missing return statement
		pass


class AsyncOpenAIMockError(AsyncOpenAIMockBase):
	async def create(self, **kwargs: Any) -> ChatCompletion:  # noqa: ARG002
		raise OpenAIError("Mocked OpenAIError")


class AsyncOpenAIMockResponse(AsyncOpenAIMockBase):
	async def create(self, **kwargs: Any) -> ChatCompletion:  # noqa: ARG002
		return ChatCompletion(
			id="chatcmpl-54IaqQSHJIRyelWm0XDbvyVvrWkFG",
			choices=[
				Choice(
					finish_reason="stop",
					index=0,
					logprobs=None,
					message=ChatCompletionMessage(
						content="This is a test.",
						role="assistant",
						function_call=None,
						tool_calls=None,
					),
				)
			],
			created=1711526796,
			model="gpt-3.5-turbo-0125",
			object="chat.completion",
			system_fingerprint="fp_3bc1b5746c",
			usage=CompletionUsage(
				completion_tokens=5, prompt_tokens=12, total_tokens=17
			),
		)


@patch("martian.anthropic.AsyncAnthropic", AsyncAnthropicMockResponse)
async def test_get_llm_message_anthropic(client: AsyncClient) -> None:
	response = await client.get(
		"/anthropic",
		params={
			"max_tokens": 50,
			"string_messages": '[{"role": "user", "content": "Hello, Claude"}]',
			"model": "claude-3-opus-20240229",
		},
	)
	assert response.status_code == status.HTTP_200_OK


@patch("martian.anthropic.AsyncAnthropic", AsyncAnthropicMockError)
async def test_get_llm_message_anthropic_error(client: AsyncClient) -> None:
	response = await client.get(
		"/anthropic",
		params={
			"max_tokens": 50,
			"string_messages": '[{"role": "user", "content": "Hello, Claude"}]',
			"model": "bad_model",
		},
	)
	assert response.status_code == status.HTTP_400_BAD_REQUEST


@patch("martian.openai.AsyncOpenAI", AsyncOpenAIMockResponse)
async def test_get_llm_message_openai(client: AsyncClient) -> None:
	response = await client.get(
		"/openai",
		params={
			"max_tokens": 50,
			"string_messages": (
				'[{"role": "user", "content": "Say this is a test"}]'
			),
			"model": "gpt-3.5-turbo",
		},
	)
	assert response.status_code == status.HTTP_200_OK


@patch("martian.openai.AsyncOpenAI", AsyncOpenAIMockError)
async def test_get_llm_message_openai_error(client: AsyncClient) -> None:
	response = await client.get(
		"/openai",
		params={
			"max_tokens": 50,
			"string_messages": (
				'[{"role": "user", "content": "Say this is a test"}]'
			),
			"model": "bad_model",
		},
	)
	assert response.status_code == status.HTTP_400_BAD_REQUEST
