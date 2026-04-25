"""Configuration loaded from environment / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    telegram_bot_username: str = "nodevision_bot"
    telegram_default_chat_id: str = ""

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "signals:btc:4h"
    redis_consumer_group: str = "tg_bot_v1"
    redis_consumer_name: str = "bot_main"

    postgres_url: str = "postgres://elibri:elibri@localhost:5432/elibri"

    log_level: str = "INFO"
    signal_format: str = "default"
    dry_run: bool = False

    @property
    def chat_id_int(self) -> int | None:
        # Telegram channel IDs are integers like -1001234567890.
        # An empty string means "not configured yet" — bot will warn at startup.
        if not self.telegram_default_chat_id:
            return None
        return int(self.telegram_default_chat_id)


settings = Settings()  # type: ignore[call-arg]
