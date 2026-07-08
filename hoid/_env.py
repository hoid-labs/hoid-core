from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Load environment variables from the nearest .env file.

    Searches upward from the current working directory. Safe to call multiple
    times — subsequent calls are no-ops if the file has already been loaded.
    """
    load_dotenv(find_dotenv(usecwd=True), override=True)
