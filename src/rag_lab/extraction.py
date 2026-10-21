"""Document conversion is loaded only when explicitly requested."""
from pathlib import Path


def convert_document(source: str | Path) -> dict:
    try:
        from docling.document_converter import DocumentConverter
        from docling.exceptions import ConversionError
    except ImportError as error:
        raise RuntimeError('Document extraction requires: python -m pip install -e ".[documents]"') from error
    try:
        document = DocumentConverter().convert(str(source)).document
        return {**document.export_to_dict(), "markdown": document.export_to_markdown()}
    except ConversionError as error:
        raise RuntimeError(f"Could not convert the document: {source}") from error


def extract_text(source: str | Path) -> str:
    document = convert_document(source)
    if "markdown" in document:
        return document["markdown"]
    return "\n".join(block["orig"] for block in document.get("texts", []) if block.get("orig"))
