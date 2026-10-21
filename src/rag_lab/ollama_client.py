"""Keep embedding and answer-generation models separate."""
from collections.abc import Sequence

from ollama import Client, ResponseError

from rag_lab.config import Settings


def embed_texts(texts: Sequence[str], settings: Settings, *, model: str | None = None) -> list[list[float]]:
    if isinstance(texts, str) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text to embed.")
    model = model or settings.embedding_model
    try:
        with Client(host=settings.ollama_host, timeout=120) as client:
            response = client.embed(model=model, input=list(texts), truncate=False)
    except ResponseError as error:
        if error.status_code == 404:
            raise RuntimeError(f"Embedding model {model!r} is unavailable. Download it with ollama pull.") from error
        raise RuntimeError(str(error)) from error
    return response.embeddings


def generate_answer(prompt: str, settings: Settings, *, model: str | None = None, max_tokens: int = 1024) -> str:
    if not prompt.strip() or max_tokens < 1:
        raise ValueError("Provide a nonempty prompt and a positive token limit.")
    model = model or settings.chat_model
    try:
        with Client(host=settings.ollama_host, timeout=120) as client:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens},
            )
    except ResponseError as error:
        if error.status_code == 404:
            raise RuntimeError(f"Chat model {model!r} is unavailable. Create or download it with Ollama.") from error
        raise RuntimeError(str(error)) from error
    if not response.message.content:
        raise RuntimeError("The model returned no final answer. Try a larger --max-tokens value.")
    return response.message.content
