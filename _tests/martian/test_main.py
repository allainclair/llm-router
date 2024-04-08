from asyncio import gather

from fastapi import status
from httpx import AsyncClient


async def test_get_llm_message_anthropic(client: AsyncClient) -> None:
	response = await client.get(
		"/anthropic",
		params={
			"max_tokens": 100,
			"string_messages": (
				'[{"role": "user", "content": "Give me a text example"}]'
			),
			"model": "claude-3-opus-20240229",
		},
	)
	# print(response.json())  # Print the response to see the result.
	assert response.status_code == status.HTTP_200_OK


async def test_get_llm_message_openai(client: AsyncClient) -> None:
	response = await client.get(
		"/openai",
		params={
			"max_tokens": 100,
			"string_messages": (
				'[{"role": "user", "content": "Give me a text example"}]'
			),
			"model": "gpt-3.5-turbo",
			"stream": "true",
		},
	)
	assert response.status_code == status.HTTP_200_OK


async def test_get_llm_message_together(client: AsyncClient) -> None:
	response = await client.get(
		"/together",
		params={
			"base_url": "https://api.together.xyz/v1",
			"string_messages": (
				'[{"role": "user", "content": "Give me a text example"}]'
			),
			"model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
		},
	)
	assert response.status_code == status.HTTP_200_OK
	# print(response.json())  # Print the response to see the result.


async def test_get_llm_message_openai_many_requests(
	client: AsyncClient,
) -> None:
	n_requests = 1000
	tasks = []
	for _ in range(n_requests):
		tasks = [
			client.get(
				"/openai",
				params={
					"max_tokens": 100,
					"string_messages": (
						'[{"role": "user", "content": "Give me text example"}]'
					),
					"model": "gpt-3.5-turbo",
				},
				timeout=60,
			)
			for _ in range(n_requests)
		]
	responses = await gather(*tasks)
	assert len(responses) == n_requests
	for response in responses:
		# See the time it took to get the response.
		# print(response.elapsed.total_seconds())
		assert response.status_code == status.HTTP_200_OK


async def test_get_many_llm_messages_from_all_llm_services(
	client: AsyncClient,
) -> None:
	n_requests_per_llm_service = 200
	tasks = []

	# Anthropic
	for _ in range(n_requests_per_llm_service):
		tasks.append(
			client.get(
				"/anthropic",
				params={
					"max_tokens": 100,
					"string_messages": (
						'[{"role": "user", "content": "Give me text example"}]'
					),
					"model": "claude-3-opus-20240229",
				},
				timeout=60,
			)
		)

	# OpenAI
	for _ in range(n_requests_per_llm_service):
		tasks.append(
			client.get(
				"/openai",
				params={
					"max_tokens": 100,
					"string_messages": (
						'[{"role": "user", "content": "Give me text example"}]'
					),
					"model": "gpt-3.5-turbo",
				},
				timeout=60,
			)
		)

	# Together
	for _ in range(n_requests_per_llm_service):
		tasks.append(
			client.get(
				"/together",
				params={
					"max_tokens": 100,
					"string_messages": (
						'[{"role": "user", "content": "Give me text example"}]'
					),
					"model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
					"base_url": "https://api.together.xyz/v1",
				},
				timeout=60,
			)
		)

	responses = await gather(*tasks)
	assert len(responses) == n_requests_per_llm_service * 3
	for response in responses:
		# See the time it took to get the response.
		# print(response.elapsed.total_seconds())
		assert response.status_code == status.HTTP_200_OK


async def test_get_llm_message_openai_to_the_http_server() -> None:
	"""This needs the server running: pdm run uvicorn martian.main:app"""
	async with AsyncClient(base_url="http://0.0.0.0:8000") as client:
		await test_get_llm_message_openai_many_requests(client)
