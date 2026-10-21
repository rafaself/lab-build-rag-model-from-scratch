"""Download a resumable Wikipedia corpus, retaining article and failure metadata."""
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
import time

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusConfig:
    output_dir: Path
    contact: str
    search_term: str = "astronomy"
    num_articles: int = 100
    language: str = "en"
    request_delay: float = 2.0
    max_retries: int = 5
    timeout: float = 30.0

    def __post_init__(self):
        if self.num_articles < 1 or self.max_retries < 1:
            raise ValueError("Article count and retry count must be positive.")
        if self.timeout <= 0 or self.request_delay < 0:
            raise ValueError("Timeout must be positive and request delay cannot be negative.")
        if not self.contact.strip() or "\n" in self.contact or "\r" in self.contact:
            raise ValueError("Set WIKIPEDIA_CONTACT in .env or pass --contact to identify API requests.")
        if not self.search_term.strip():
            raise ValueError("The search topic cannot be empty.")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", self.language):
            raise ValueError("Use a Wikipedia language code such as en, pt, or simple.")


def sanitize_filename(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_-]", "_", text)).strip("_")[:100] or "untitled"


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read {path}; inspect the file before continuing.") from error
    if not isinstance(records, list) or any(not isinstance(record, dict) or "page_id" not in record for record in records):
        raise ValueError(f"Invalid corpus records in {path}.")
    return records


def atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_records(path: Path, records: list[dict]) -> None:
    atomic_write(path, json.dumps(records, indent=4, ensure_ascii=False) + "\n")


def article_path(output_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or filename in (".", ".."):
        raise ValueError("Corpus metadata must use plain article filenames.")
    return output_dir / filename


class WikipediaClient:
    def __init__(self, config: CorpusConfig):
        self.config = config
        self.api_url = f"https://{config.language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"AstronomyCorpusResearch/1.0 ({config.contact}) requests",
            "Accept": "application/json",
        })

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.session.close()

    def request(self, params: dict) -> dict:
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            # Never reuse a response from an earlier failed attempt.
            response = None
            try:
                response = self.session.get(self.api_url, params=params, timeout=self.config.timeout)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    response.raise_for_status()
                if response.status_code != 200:
                    raise RuntimeError(f"Wikipedia returned HTTP {response.status_code}.")
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Wikipedia returned an invalid JSON response shape.")
                if "error" in data:
                    raise RuntimeError(f"Wikipedia API error: {data['error']}")
                return data
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as error:
                last_error = error
                if attempt == self.config.max_retries:
                    break
                delay = min(60, 2 ** attempt) + random.uniform(0, 1)
                if response is not None and response.status_code == 429:
                    retry_after = retry_after_seconds(response.headers.get("Retry-After"))
                    if retry_after is not None:
                        delay = retry_after
                logger.warning("Request failed (%s/%s); retrying in %.1f seconds.", attempt, self.config.max_retries, delay)
                time.sleep(delay)
            finally:
                time.sleep(self.config.request_delay)
        raise RuntimeError(f"Wikipedia request failed after {self.config.max_retries} attempts.") from last_error

    def search(self) -> list[dict]:
        results = []
        seen = set()
        continuation = {}
        while len(results) < self.config.num_articles:
            data = self.request({
                "action": "query", "list": "search",
                "srsearch": self.config.search_term.replace("_", " "),
                "srlimit": min(50, self.config.num_articles - len(results)),
                "format": "json", "formatversion": "2", **continuation,
            })
            batch = data.get("query", {}).get("search", [])
            if not batch:
                break
            for article in batch:
                if str(article["pageid"]) not in seen:
                    seen.add(str(article["pageid"]))
                    results.append(article)
            logger.info("Search progress: %s/%s", len(results), self.config.num_articles)
            next_page = data.get("continue", {})
            if not next_page:
                break
            if next_page == continuation:
                raise RuntimeError("Wikipedia returned a pagination cursor that did not advance.")
            continuation = next_page
        return results[:self.config.num_articles]

    def fetch(self, page_id: str) -> dict:
        data = self.request({
            "action": "query", "prop": "extracts|info", "pageids": page_id,
            "explaintext": "1", "inprop": "url", "format": "json", "formatversion": "2",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise RuntimeError("Wikipedia article does not exist.")
        return pages[0]


def generate_corpus(config: CorpusConfig) -> dict:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    failures_path = output_dir / "failed_articles.json"
    metadata = load_records(metadata_path)
    failures = load_records(failures_path)
    saved_ids = {
        str(record["page_id"]) for record in metadata
        if article_path(output_dir, record["filename"]).is_file()
    }
    successful = skipped = errors = 0
    with WikipediaClient(config) as client:
        results = client.search()
        for result in results:
            page_id = str(result["pageid"])
            if page_id in saved_ids:
                skipped += 1
                logger.info("Already saved: %s", result["title"])
                continue
            try:
                article = client.fetch(page_id)
                content = article.get("extract", "").strip()
                if not content:
                    raise RuntimeError("Empty article content.")
            except (requests.RequestException, RuntimeError, ValueError) as error:
                errors += 1
                failures = [record for record in failures if str(record["page_id"]) != page_id]
                failures.append({"page_id": page_id, "title": result["title"], "error": str(error)})
                save_records(failures_path, failures)
                logger.error("Could not save %s: %s", result["title"], error)
                continue
            filename = f"{page_id}_{sanitize_filename(article['title'])}.txt"
            atomic_write(article_path(output_dir, filename), content)
            item = {
                "page_id": page_id, "title": article["title"], "url": article.get("fullurl"),
                "filename": filename, "word_count": len(content.split()), "character_count": len(content),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata = [record for record in metadata if str(record["page_id"]) != page_id] + [item]
            save_records(metadata_path, metadata)
            failures = [record for record in failures if str(record["page_id"]) != page_id]
            save_records(failures_path, failures)
            saved_ids.add(page_id)
            successful += 1
            logger.info("Saved: %s (%s words)", article["title"], item["word_count"])
    return {
        "search_results": len(results), "saved": successful, "skipped": skipped, "failed": errors,
        "total_metadata": len(metadata),
        "total_words": sum(record["word_count"] for record in metadata if article_path(output_dir, record["filename"]).is_file()),
        "output_dir": str(output_dir),
    }
