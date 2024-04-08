from functools import lru_cache

from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from martian.enums import LLMServiceEnum
from martian.exceptions import http_400_invalid_llm_service


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env", env_file_encoding="utf-8"
	)

	api_key_anthropic: SecretStr = SecretStr("")
	api_key_openai: SecretStr = SecretStr("")
	api_key_together: SecretStr = SecretStr("")

	base_url_together: HttpUrl = HttpUrl("https://api.together.xyz/v1")


@lru_cache()
def get_settings() -> Settings:
	return Settings()


def get_api_key(llm_service: LLMServiceEnum) -> SecretStr:
	settings = get_settings()
	match llm_service:
		case LLMServiceEnum.ANTHROPIC:
			return settings.api_key_anthropic
		case LLMServiceEnum.OPENAI:
			return settings.api_key_openai
		case LLMServiceEnum.TOGETHER:
			return settings.api_key_together
		case _:
			raise http_400_invalid_llm_service(llm_service)
