# Environment Variables

The `hoid` relies on environment variables for configuration. To set up your local environment, copy the `.env.example` file to a new file named `.env` and fill in your specific values.

The table below is generated from `.env.example`, so it stays in sync with the codebase.
Use it as the source of truth when configuring local runs, examples, or MCP/plugin servers.

## How to use this reference

- Start with required core variables (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`).
- Add feature-specific variables only for the extras you installed.
- Keep provider credentials in `.env` and out of source-controlled files.

## Variable groups

- **Core model access**: base URL, API key, model name.
- **RAG/vector settings**: embedding model and backend-specific options.
- **Auth/OIDC settings**: identity provider parameters for web auth flows.
- **Plugin/server settings**: service credentials used by MCP/plugin adapters.

Below is a complete reference of all available configuration options:

{{ env_table() }}

