import json
import logging
import re
import zipfile
from pathlib import Path

import defusedxml.ElementTree as ET

from hoid._optional import require as _require

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# maps OOXML paragraph style names to markdown heading prefixes
HEADING_STYLES = {
    "Heading1": "#",
    "Heading2": "##",
    "Heading3": "###",
    "Heading4": "####",
    "Heading5": "#####",
    "Heading6": "######",
}


def to_markdown(path: Path) -> str:
    "Convert a file to markdown-structured text for chunking and embedding."
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".rst", ".csv", ".json", ".jsonl"}:
        return path.read_text(errors="replace")

    if suffix in {".html", ".htm"}:
        text = path.read_text(errors="replace")
        text = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if suffix == ".xml":
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            if root is None:
                return ""
            return " ".join(root.itertext())
        except ET.ParseError as exc:
            log.warning("XML parse error in %s: %s", path.name, exc)
            return ""

    if suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(errors="replace"))
            lang = (
                nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
            )
            parts = []
            for cell in nb.get("cells", []):
                src = cell.get("source", "")
                text = "".join(src) if isinstance(src, list) else src
                if not text.strip():
                    continue
                if cell.get("cell_type") == "markdown":
                    parts.append(text)
                else:
                    parts.append(f"```{lang}\n{text}\n```")
            return "\n\n".join(parts)
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Notebook parse error in %s: %s", path.name, exc)
            return ""

    if suffix == ".docx":
        try:
            with zipfile.ZipFile(path) as zf:
                xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            parts = []
            for para in root.findall(".//{*}p"):
                style_elem = para.find(".//{*}pStyle")
                if style_elem is not None:
                    # attribute name is namespace-qualified; match by suffix for robustness
                    style = next(
                        (
                            v
                            for k, v in style_elem.attrib.items()
                            if k.endswith("}val") or k == "val"
                        ),
                        "",
                    )
                else:
                    style = ""
                text = "".join(t.text or "" for t in para.findall(".//{*}t"))
                if not text.strip():
                    continue
                prefix = HEADING_STYLES.get(style, "")
                parts.append(f"{prefix} {text}" if prefix else text)
            return "\n\n".join(parts)
        except (KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
            log.warning("DOCX parse error in %s: %s", path.name, exc)
            return ""

    if suffix == ".pdf":
        _require("pypdf", pypdf)
        reader = pypdf.PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    # unknown format: attempt plain text, return empty on decode failure
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""
