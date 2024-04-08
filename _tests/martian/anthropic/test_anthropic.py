from dotenv import dotenv_values
from rich import print

from martian.anthropic import AsyncAnthropic


async def test_main() -> None:
	env = dotenv_values(".env")
	client = AsyncAnthropic(api_key=env.get("API_KEY_ANTHROPIC"))
	message = await client.messages.create(
		model="claude-3-opus-20240229",
		max_tokens=1024,
		messages=[
			{
				"role": "user",
				"content": "Hello, Claude, give me a text example.",
			}
		],
		stream=True,
	)
	text = ""
	async for chunk in message:
		print(chunk)
	# if chunk.contents[0].content:
	# 	text += chunk.contents[0].content
	assert text

	print(message)
