import json
from typing import Any

from fastapi import Query
from pydantic import BaseModel, HttpUrl, computed_field

from martian.exceptions import http_400_invalid_string_messages_field

MIN_LIMIT = 1
MAX_LIMIT = 1_000
MAX_STRING_MESSAGE = 10_000


class RequestQuery(BaseModel):
	base_url: HttpUrl | None = Query(
		default=None,
		description="Base url for the LLM service.",
	)
	max_tokens: int | None = Query(
		default=None,
		ge=MIN_LIMIT,
		le=MAX_LIMIT,
		description="The maximum number of tokens to generate.",
	)
	model: str | None = Query(
		default=None,
		min_length=MIN_LIMIT,
		max_length=MAX_LIMIT,
		description="Model of to the LLM service",
	)
	stream: bool = Query(default=False, description="Stream the response.")
	string_messages: str = Query(
		default=None,
		min_length=MIN_LIMIT,
		max_length=MAX_STRING_MESSAGE,
		description="A list of messages to generate responses for.",
	)

	@computed_field  # type: ignore[misc]
	@property
	def messages(self) -> list[dict[Any, Any]]:
		try:
			messages_: list[dict[Any, Any]] = json.loads(self.string_messages)
			return messages_
		except json.JSONDecodeError:
			raise http_400_invalid_string_messages_field(self.string_messages)
