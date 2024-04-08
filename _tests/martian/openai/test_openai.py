from dotenv import dotenv_values
from rich import print

from martian.openai import AsyncOpenAI


async def test_main() -> None:
	env = dotenv_values(".env")
	client = AsyncOpenAI(api_key=env.get("API_KEY_OPENAI"))
	message = await client.chat.completions.create(
		messages=[
			{
				"role": "user",
				"content": "Give me a text example to test a text stream",
			}
		],
		model="gpt-3.5-turbo",
		# stream=True,
	)
	print(message)
	# cat = ""
	# async for chunk in message:
	# 	print(chunk)
	# 	if chunk.choices[0].delta.content:
	# 		cat += chunk.choices[0].delta.content
	# print(cat)

	# response = openai.Completion.create(
	# 	engine="gpt-3.5-turbo",
	# 	prompt="What is the capital of France?",
	# 	max_tokens=50,
	# 	temperature=0.7,
	# 	top_p=1,
	# 	frequency_penalty=0,
	# 	presence_penalty=0
	# )
	# print(response.choices[0].text.strip())
