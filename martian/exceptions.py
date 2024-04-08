from fastapi import HTTPException, status


def http_400_invalid_llm_service(llm_service: str) -> HTTPException:
	return HTTPException(
		status_code=status.HTTP_400_BAD_REQUEST,
		detail=f"Invalid LLM service: {llm_service=}",
	)


def http_400_invalid_string_messages_field(
	string_messages: str,
) -> HTTPException:
	return HTTPException(
		status_code=status.HTTP_400_BAD_REQUEST,
		detail=f"Invalid JSON from field string_messages: '{string_messages}'",
	)


def http_400_llm_service_error(error: str) -> HTTPException:
	return HTTPException(
		status_code=status.HTTP_400_BAD_REQUEST,
		detail=f"LLM service error {error}:",
	)
