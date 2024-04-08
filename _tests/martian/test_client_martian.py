"""End-to-end tests for AsyncMartian to LLN services.

To run the tests:
$ pdm run pytest -ssvv
	_tests/martian/test_client_martian.py  # Test all

# Test individual test
$ pdm run pytest -ssvv
	_tests/martian/test_client_martian.py::TestAsyncMartian::<chosen_test>
"""

from martian.client_martian import AsyncMartian
from martian.enums import LLMServiceEnum
from martian.models.martian import Message
from martian.settings import get_api_key


class TestAsyncMartian:
	async def test_create_anthropic(self) -> None:
		client = AsyncMartian(
			llm_service=LLMServiceEnum.ANTHROPIC,
			api_key=get_api_key(LLMServiceEnum.ANTHROPIC).get_secret_value(),
		)
		response = await client.create(
			model="claude-3-opus-20240229",
			max_tokens=1024,
			messages=[{"role": "user", "content": "Hello, Claude"}],
		)
		# print(response.json())  # Print the response to see the result.
		assert response

	async def test_create_anthropic_stream(self) -> None:
		client = AsyncMartian(
			llm_service=LLMServiceEnum.ANTHROPIC,
			api_key=get_api_key(LLMServiceEnum.ANTHROPIC).get_secret_value(),
		)
		response = await client.create(
			model="claude-3-opus-20240229",
			max_tokens=1024,
			messages=[{"role": "user", "content": "Give me a text example."}],
			stream=True,
		)
		assert response
		assert not isinstance(response, Message)

		text = ""
		async for chunk in response:
			if chunk.contents and chunk.contents[0].content:
				text += chunk.contents[0].content
		# print(text)  # Print text if you want to see the result
		assert text

	async def test_create_openai_stream(self) -> None:
		client = AsyncMartian(
			llm_service=LLMServiceEnum.OPENAI,
			api_key=get_api_key(LLMServiceEnum.OPENAI).get_secret_value(),
		)
		response = await client.create(
			messages=[
				{
					"role": "user",
					"content": "Give ma text example to test",
				}
			],
			model="gpt-3.5-turbo",
			stream=True,
		)
		assert response
		assert not isinstance(response, Message)

		text = ""
		async for chunk in response:
			if chunk.contents[0].content:
				text += chunk.contents[0].content
		# print(text)  # Print text if you want to see the result
		assert text

	async def test_create_together(self) -> None:
		client = AsyncMartian(
			llm_service=LLMServiceEnum.TOGETHER,
			api_key=get_api_key(LLMServiceEnum.TOGETHER).get_secret_value(),
			base_url="https://api.together.xyz/v1",
		)
		response = await client.create(
			model="mistralai/Mixtral-8x7B-Instruct-v0.1",
			messages=[
				{
					"role": "system",
					"content": "You are an expert travel guide.",
				},
				{
					"role": "user",
					"content": "Tell me fun things to do in San Francisco.",
				},
			],
		)
		# print(response)  # Print the response if you want to see the result.
		assert response

	async def test_create_together_stream(self) -> None:
		client = AsyncMartian(
			llm_service=LLMServiceEnum.TOGETHER,
			api_key=get_api_key(LLMServiceEnum.TOGETHER).get_secret_value(),
			base_url="https://api.together.xyz/v1",
		)
		response = await client.create(
			model="mistralai/Mixtral-8x7B-Instruct-v0.1",
			messages=[
				{
					"role": "system",
					"content": "You are an expert travel guide.",
				},
				{
					"role": "user",
					"content": "Tell me fun things to do in San Francisco.",
				},
			],
			stream=True,
		)
		assert response
		assert not isinstance(response, Message)

		text = ""
		async for chunk in response:
			if chunk.contents[0].content:
				text += chunk.contents[0].content
		# print(text)  # Print text if you want to see the result
		assert text
