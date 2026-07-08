import asyncio
import ipaddress
import re
import socket
import urllib.parse

import httpx

from hoid.core import tool

# cap to avoid flooding context
_MAX_CHARS = 12_000

# scheme check alone isn't sufficient; resolve and block private/reserved ranges to prevent SSRF
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _strip_html(html):
    # strip script/style before the generic pattern or their body text leaks through
    html = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s{2,}", " ", html).strip()


async def _assert_not_ssrf(url: str) -> None:
    hostname = urllib.parse.urlparse(url).hostname
    if not hostname:
        raise ValueError(f"Could not parse hostname from URL: {url}")
    try:
        records = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve '{hostname}': {exc}") from exc
    for _, _, _, _, addr in records:
        if any(ipaddress.ip_address(addr[0]) in net for net in _BLOCKED_NETWORKS):
            raise ValueError(
                f"Blocked: '{url}' resolves to a private or reserved address"
            )


@tool
async def fetch_url(url: str, as_text: bool = True) -> str:
    """Fetch a URL and return its content as text.

    Args:
        url: The URL to fetch (http/https only).
        as_text: If True, strip HTML tags from HTML responses.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("Only http/https URLs are allowed")
    await _assert_not_ssrf(url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.text
    # skip stripping for non-HTML content types
    if as_text and "html" in r.headers.get("content-type", ""):
        body = _strip_html(body)
    return body[:_MAX_CHARS] + ("\n...[truncated]" if len(body) > _MAX_CHARS else "")
