"""Run each lab exercise explicitly from one small command-line interface."""
import argparse
import json
import logging
from pathlib import Path

from rag_lab.config import Settings
from rag_lab.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive integer.") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-lab", description="Run the local RAG lab exercises.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("db-check", help="Check the configured PostgreSQL database and pgvector.")
    embed = commands.add_parser("embed", help="Embed one or more texts with Ollama.")
    embed.add_argument("texts", nargs="+")
    embed.add_argument("--model", help="Override OLLAMA_EMBED_MODEL.")
    chat = commands.add_parser("chat", help="Generate an answer with Ollama.")
    chat.add_argument("prompt")
    chat.add_argument("--model", help="Override OLLAMA_CHAT_MODEL.")
    chat.add_argument("--max-tokens", type=positive_integer, default=1024)
    extract = commands.add_parser("extract", help="Extract text from a document path or URL.")
    extract.add_argument("source")
    extract.add_argument("--output", type=Path, help="Write the text to a file instead of stdout.")
    corpus = commands.add_parser("corpus", help="Download or resume a Wikipedia article corpus.")
    corpus.add_argument("--topic", default="astronomy")
    corpus.add_argument("--count", type=positive_integer, default=100)
    corpus.add_argument("--output-dir", type=Path)
    corpus.add_argument("--language", default="en")
    corpus.add_argument("--contact", help="Override WIKIPEDIA_CONTACT.")
    populate = commands.add_parser("populate", help="Store local text chunks and embeddings in PostgreSQL.")
    populate.add_argument("--input-dir", type=Path, help="Directory of .txt files; defaults to CORPUS_DIR.")
    populate.add_argument("--model", help="Override OLLAMA_EMBED_MODEL.")
    populate.add_argument("--chunk-size", type=positive_integer, default=DEFAULT_CHUNK_SIZE, help=f"Tokens per chunk (default: {DEFAULT_CHUNK_SIZE}).")
    populate.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help=f"Target overlapping tokens (default: {DEFAULT_OVERLAP}).")
    populate.add_argument("--strategy", choices=("tokens", "words"), default="tokens", help="Use tokens (default), or reproduce the legacy word-window baseline.")
    populate.add_argument("--batch-size", type=positive_integer, default=16, help="Chunks per embedding request (default: 16).")
    populate.add_argument("--limit", type=positive_integer, help="Process only the first N text files, sorted by path.")
    populate.add_argument("--force", action="store_true", help="Re-embed unchanged files for this model.")
    pull = commands.add_parser("pull", help="Retrieve the closest text chunks from the vector database.")
    pull.add_argument("query", help="Text to search for semantically.")
    pull.add_argument("--top-k", type=positive_integer, default=5, help="Maximum chunks to return (default: 5).")
    pull.add_argument("--model", help="Override OLLAMA_EMBED_MODEL; must match the stored model.")
    pull.add_argument("--min-similarity", type=float, help="Minimum raw cosine similarity (-1 to 1).")
    pull.add_argument("--candidate-k", type=positive_integer, help="Vector candidates to rerank; at most 1000.")
    pull.add_argument("--no-deduplicate", action="store_true")
    pull.add_argument("--no-rerank", action="store_true")
    ask = commands.add_parser("ask", help="Answer a question using retrieved chunks with source references.")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=positive_integer, default=3)
    ask.add_argument("--embedding-model", help="Override OLLAMA_EMBED_MODEL.")
    ask.add_argument("--chat-model", help="Override OLLAMA_CHAT_MODEL.")
    ask.add_argument("--max-tokens", type=positive_integer, default=2048)
    ask.add_argument("--max-context-chars", type=positive_integer, default=6000)
    ask.add_argument("--min-similarity", type=float, default=0.3)
    ask.add_argument("--candidate-k", type=positive_integer)
    ask.add_argument("--no-deduplicate", action="store_true")
    ask.add_argument("--no-rerank", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Compare baseline and improved retrieval on labeled questions.")
    evaluate.add_argument("--dataset", type=Path)
    evaluate.add_argument("--min-similarity", type=float, default=0.3)
    evaluate.add_argument("--candidate-k", type=positive_integer, default=40)
    evaluate.add_argument("--output", type=Path, help="Save the JSON report instead of printing it.")
    tune = commands.add_parser("tune-chunking", help="Compare token chunk sizes in temporary PostgreSQL schemas.")
    tune.add_argument("--dataset", type=Path)
    tune.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "db-check":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.database import check_database
        try:
            print(json.dumps(check_database(settings), indent=2))
        except SQLAlchemyError as error:
            raise RuntimeError("Database connection failed. Check .env and docker compose ps; pgvector must be enabled.") from error
    elif args.command == "embed":
        from rag_lab.ollama_client import embed_texts
        vectors = embed_texts(args.texts, settings, model=args.model)
        print(json.dumps({"model": args.model or settings.embedding_model, "count": len(vectors),
                          "dimensions": len(vectors[0]) if vectors else 0}, indent=2))
    elif args.command == "chat":
        from rag_lab.ollama_client import generate_answer
        print(generate_answer(args.prompt, settings, model=args.model, max_tokens=args.max_tokens))
    elif args.command == "extract":
        from rag_lab.extraction import extract_text
        content = extract_text(args.source)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            print(f"Saved extracted text to {args.output}")
        else:
            print(content)
    elif args.command == "tune-chunking":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.evaluate import DEFAULT_DATASET
        from rag_lab.tune_chunking import tune_chunking
        try:
            result = tune_chunking(settings, dataset=args.dataset or DEFAULT_DATASET)
        except SQLAlchemyError as error:
            raise RuntimeError("Chunk tuning failed. Check PostgreSQL and schema creation permissions.") from error
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved chunking comparison to {args.output}")
    elif args.command == "evaluate":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.evaluate import DEFAULT_DATASET, evaluate_retrieval
        try:
            result = evaluate_retrieval(settings, dataset=args.dataset or DEFAULT_DATASET,
                                        min_similarity=args.min_similarity, candidate_k=args.candidate_k)
        except SQLAlchemyError as error:
            raise RuntimeError("Evaluation could not read the vector database. Check PostgreSQL and populate first.") from error
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(f"Saved evaluation to {args.output}")
        else:
            print(serialized, end="")
    elif args.command == "ask":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.ask import ask_question
        try:
            result = ask_question(
                args.question, settings, top_k=args.top_k, embedding_model=args.embedding_model,
                chat_model=args.chat_model, max_tokens=args.max_tokens, max_context_chars=args.max_context_chars,
                min_similarity=args.min_similarity, candidate_k=args.candidate_k,
                deduplicate=not args.no_deduplicate, rerank=not args.no_rerank,
            )
        except SQLAlchemyError as error:
            raise RuntimeError("RAG retrieval failed. Check PostgreSQL and run rag-lab populate first.") from error
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "pull":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.pull import pull_chunks
        try:
            matches = pull_chunks(args.query, settings, top_k=args.top_k, model=args.model,
                                  min_similarity=args.min_similarity, candidate_k=args.candidate_k,
                                  deduplicate=not args.no_deduplicate, rerank=not args.no_rerank)
        except SQLAlchemyError as error:
            raise RuntimeError("Database retrieval failed. Check PostgreSQL and .env, then run rag-lab populate if the tables are missing.") from error
        print(json.dumps({"query": args.query, "model": args.model or settings.embedding_model,
                          "matches": matches}, indent=2, ensure_ascii=False))
    elif args.command == "populate":
        from sqlalchemy.exc import SQLAlchemyError
        from rag_lab.populate import populate_database
        try:
            result = populate_database(
                settings, input_dir=args.input_dir, model=args.model, chunk_size=args.chunk_size,
                overlap=args.overlap, batch_size=args.batch_size, limit=args.limit, force=args.force, strategy=args.strategy,
            )
        except SQLAlchemyError as error:
            raise RuntimeError("Database population failed. Check PostgreSQL, .env, and pgvector. The current document was rolled back; earlier documents remain saved.") from error
        print(json.dumps(result, indent=2))
    elif args.command == "corpus":
        from rag_lab.corpus import CorpusConfig, generate_corpus
        result = generate_corpus(CorpusConfig(
            output_dir=args.output_dir or settings.corpus_dir, contact=args.contact or settings.wikipedia_contact,
            search_term=args.topic, num_articles=args.count, language=args.language,
        ))
        print(json.dumps(result, indent=2))
        return 1 if result["failed"] else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return run(args, Settings.from_env())
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
