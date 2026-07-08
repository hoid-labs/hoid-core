# Client Construction

`LLMClient.from_env()` reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and `CA_BUNDLE_PATH` from the environment (`.env` file loaded automatically) and constructs the client.

```python
from hoid.core import LLMClient

async with LLMClient.from_env() as client:
    ...
```

To use multiple endpoints in the same script, construct `LLMClient` directly:

```python
local = LLMClient(base_url="http://localhost:1234/v1", api_key="nokey", model="qwen3-4b")
remote = LLMClient(base_url="https://api.example.com", api_key="sk-...", model="gpt-4o")
```

### ::: hoid.core.llm.LLMClient