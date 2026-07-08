# Contributing

Thanks for your interest in `hoid`. This document covers the mechanics of contributing — for design rules, code style, and architectural guidance, see [`CLAUDE.md`](CLAUDE.md) at the repo root. AI agents (Claude Code, GitHub Copilot, Cursor, etc.) working in this repo also read `CLAUDE.md`; keep it current.

## Setup

```sh
git clone https://github.com/hoid-labs/hoid-core
cd hoid-core
uv venv
source .venv/bin/activate
uv pip install -e ".[std]"   # core + rag + oidc
uv sync --all-extras --group dev
```

## Running the tests

Unit tests run without a live LLM endpoint. Integration tests require `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in your environment (see `.env.example`).

```sh
# Unit tests
uv run pytest tests/unit tests/test_packaging.py -v

# Integration tests (requires live LLM)
uv run pytest tests/integration -v -m integration

# Packaging sanity (asserts every declared extra is mentioned in the docs)
uv run --with pytest pytest tests/test_packaging.py
```

## Before you open a pull request

1. `uv run pytest tests/unit tests/test_packaging.py` — all green
2. `uv run mypy hoid` — clean
3. `uv run ruff check hoid tests` — clean
4. `uv run bandit -r hoid -ll` — no new findings
5. `uv run deptry hoid tests` — no undeclared or unused imports
6. New code is covered by tests; coverage is not reduced
7. `CHANGELOG.md` `[Unreleased]` has an entry under the right section (`Added` / `Changed` / `Fixed` / `Removed` / `Security`)
8. `CLAUDE.md` Architecture map is updated if you added files to tracked directories
9. `docs/` is updated for any user-facing change
10. The PR description links to the relevant issue, if any

CI runs the same checks on every push to `main` and on every pull request. A PR that fails CI will not be merged.

## Commit format

```
type(scope): lowercase imperative subject under 72 chars

- bullet naming what changed
```

- **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
- **Scope:** a sub-area of the codebase (`agent`, `mcp`, `rag`, `observability`, `ci`, `repo`, `docs`).
- **Subject:** conceptual only — no file paths, function names, class names, or symbol names. The body is where specifics go.
- **Body:** omit when the subject is self-explanatory; otherwise, short noun-phrase bullets naming what changed.
- **One logical change per commit.** Never bundle unrelated changes.

Examples in the git log show the convention in practice.

## Reporting bugs

Open a GitHub issue with a minimal reproduction, the output of `uv run python -V`, and the output of `uv pip show hoid`. If the bug is security-related, **do not** open a public issue — see [`SECURITY.md`](SECURITY.md) for the disclosure process.

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, assume good faith, and focus on the work.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
