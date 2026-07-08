import io
import json
import zipfile

from hoid.extensions.rag._converter import to_markdown


def test_plain_text(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("Hello plain text.")
    result = to_markdown(p)
    print(f"plain text result[:80]:\n{result[:80]}")
    assert "Hello plain text." in result


def test_markdown_passthrough(tmp_path):
    p = tmp_path / "file.md"
    p.write_text("# Title\n\nParagraph content.")
    result = to_markdown(p)
    print(f"markdown result[:80]:\n{result[:80]}")
    assert "# Title" in result
    assert "Paragraph content" in result


def test_html_strips_script_and_preserves_content(tmp_path):
    p = tmp_path / "file.html"
    p.write_text(
        "<html><body><h1>Title</h1><script>var x=1;</script><p>Body text</p>"
        "<ul><li>Item</li></ul></body></html>"
    )
    result = to_markdown(p)
    print(f"html result[:200]:\n{result[:200]}")
    assert "Title" in result
    assert "Body text" in result
    assert "Item" in result
    assert "var x" not in result


def test_xml_text_extracted(tmp_path):
    p = tmp_path / "file.xml"
    p.write_text(
        '<?xml version="1.0"?><root><item>Value A</item><item>Value B</item></root>'
    )
    result = to_markdown(p)
    print(f"xml result[:200]:\n{result[:200]}")
    assert "Value A" in result
    assert "Value B" in result


def test_ipynb_cell_types(tmp_path):
    nb = {
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Notebook Title\n", "Some explanation."],
            },
            {"cell_type": "code", "source": "print(42)"},
        ],
    }
    p = tmp_path / "file.ipynb"
    p.write_text(json.dumps(nb))
    result = to_markdown(p)
    print(f"ipynb result[:300]:\n{result[:300]}")
    assert "# Notebook Title" in result
    assert "Some explanation" in result
    assert "```python" in result
    assert "print(42)" in result


def _make_docx(tmp_path, doc_xml: str):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    p = tmp_path / "file.docx"
    p.write_bytes(buf.getvalue())
    return p


def test_docx_plain_paragraph(tmp_path):
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = f"""<?xml version="1.0"?>
<w:document xmlns:w="{W}">
  <w:body><w:p><w:r><w:t>Simple paragraph text.</w:t></w:r></w:p></w:body>
</w:document>"""
    result = to_markdown(_make_docx(tmp_path, xml))
    print(f"docx plain result:\n{result}")
    assert "Simple paragraph text" in result


def test_docx_headings(tmp_path):
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = f"""<?xml version="1.0"?>
<w:document xmlns:w="{W}">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Section A</w:t></w:r></w:p>
    <w:p><w:r><w:t>Body text here.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    result = to_markdown(_make_docx(tmp_path, xml))
    print(f"docx headings result:\n{result}")
    assert "# Chapter One" in result
    assert "## Section A" in result
    assert "Body text here" in result


def test_unknown_extension_fallback(tmp_path):
    p = tmp_path / "file.log"
    p.write_text("Log line one.\nLog line two.")
    result = to_markdown(p)
    print(f"unknown ext result:\n{result}")
    assert "Log line" in result
