# Security Policy

## Supported versions

The latest minor release of `hoid` receives security updates. Earlier minor versions are not patched; please upgrade.

| Version | Supported |
| --- | --- |
| Latest minor (`0.x` where `x` is the highest released) | yes |
| All earlier versions | no |

## Reporting a vulnerability

**Do not file a public GitHub issue for security vulnerabilities.** Public disclosure before a fix is available makes it easier for attackers to exploit the issue.

Email a description of the vulnerability to **`<security-contact>`** (replace with the address published on the project page). Include:

- A description of the vulnerability and its impact
- A minimal proof-of-concept or reproduction steps
- The version of `hoid` affected
- Your environment (Python version, OS, optional extras in use)

You can expect an acknowledgment within 3 business days. If you do not hear back, follow up on the same thread.

## Disclosure timeline

- **Day 0** — you report the issue. We acknowledge within 3 business days.
- **Day 0–14** — we investigate, develop a fix, and prepare a release. We may request additional time if the fix is non-trivial.
- **Day 14–90** — coordinated disclosure. We share the fix with you under embargo and agree on a public disclosure date. We aim to publish the fix within 90 days of the initial report.
- **After public disclosure** — the security advisory is published on GitHub, the fix is released in a patch version, and the public changelog credits you (if you wish).

We will not pursue legal action against researchers who make a good-faith effort to follow this process.

## Scope

The following are in scope:

- Code execution, sandbox escape, or arbitrary file/network access from a tool or extension where the user did not opt in
- Credential or secret disclosure through default configuration, logging, or error messages
- Bypasses of `AuthGate`, `AuthContext`, or any other access-control mechanism
- Insecure defaults shipped by the library itself (not by user code)
- Vulnerabilities in the agent loop, MCP transport, OIDC flow, or RAG store

The following are out of scope:

- Vulnerabilities in user code, user-defined tools, or user-defined guardrails
- Issues in optional third-party packages (file those upstream)
- Denial of service via malicious prompts (the model is the user; rate-limit at the deployment layer)

## Security best practices for users

- Pin `hoid` to a specific minor version in production (`hoid>=0.x,<0.(x+1)`)
- Run agents with the smallest possible tool surface — disable any tool the agent does not need
- Set `CA_BUNDLE_PATH` if your LLM endpoint uses a private CA
- Use `AuthGate` to enforce per-user tool access when serving agents to multiple users
- Subscribe to GitHub security advisories (Watch → Custom → Security alerts) to be notified of new disclosures

## Contact

**Email:** `<security-contact>` (replace with the address published on the project page)
