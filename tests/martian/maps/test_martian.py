from anthropic.types import ContentBlock
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from martian.enums import StopReasonEnum
from martian.maps.anthropic_to_martian import map_anthropic_to_martian
from martian.maps.openai_to_martian import map_openai_to_martian
from martian.models.martian import Content, Message, Usage


def test_map_anthropic_to_martian() -> None:
	anthropic_message = AnthropicMessage(
		id="msg_02WuDdq0aMLimFapk5jXWnUt",
		content=[
			ContentBlock(
				text="Hi! It's nice to meet you. How can I assist you today?",
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
	mapped_message = map_anthropic_to_martian(anthropic_message)
	assert mapped_message == Message(
		id="msg_02WuDdq0aMLimFapk5jXWnUt",
		contents=[
			Content(
				content="Hi! It's nice to meet you. How can I assist you today?"
			)
		],
		model="claude-3-opus-20240229",
		stop_reason=[StopReasonEnum.STOP],
		usage=Usage(input_tokens=10, output_tokens=19, total_tokens=29),
	)


def test_map_openai_to_martian() -> None:
	chat_completion = ChatCompletion(
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
	mapped_message = map_openai_to_martian(chat_completion)
	assert mapped_message == Message(
		id="chatcmpl-54IaqQSHJIRyelWm0XDbvyVvrWkFG",
		contents=[Content(content="This is a test.")],
		model="gpt-3.5-turbo-0125",
		stop_reason=[StopReasonEnum.STOP],
		usage=Usage(input_tokens=12, output_tokens=5, total_tokens=17),
	)
