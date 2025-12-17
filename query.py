"""
Query the brain documents vector database.

Usage:
    python query.py "search query"              # Basic search
    python query.py "query" -n 10               # Get 10 results
    python query.py "query" -t 0.5              # Filter by distance threshold
    python query.py "query" --source "*.md"     # Filter by source pattern
    python query.py "query" --latest            # Sort by latest timestamp first
    python query.py "query" --json              # Output as JSON
    python query.py --interactive               # Interactive mode
    python query.py --stats                     # Show collection stats
    python query.py --list-sources              # List all source files
"""

import argparse
import fnmatch
import json
import sys
import io
from datetime import datetime

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# Defaults
DEFAULT_DB_PATH = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "brain_docs"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_NUM_RESULTS = 5
DEFAULT_THRESHOLD = None
DEFAULT_TRUNCATE = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the brain documents vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query"
    )
    parser.add_argument(
        "--db-path", "-d",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"ChromaDB database path (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION})"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=DEFAULT_NUM_RESULTS,
        help=f"Number of results to return (default: {DEFAULT_NUM_RESULTS})"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Maximum distance threshold (filter out results above this)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Filter by source filename pattern (e.g., '*csrf*', '20251128-*.md')"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Don't truncate content in output"
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=DEFAULT_TRUNCATE,
        help=f"Truncate content to N characters (default: {DEFAULT_TRUNCATE})"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Show only metadata, no content"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive search mode"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics"
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all unique source files in the collection"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Sort results by latest timestamp first (requires re-ingestion with timestamps)"
    )

    return parser.parse_args()


def get_collection(db_path: Path, collection_name: str, model: str):
    """Get ChromaDB collection."""
    if not isinstance(db_path, Path):
        raise TypeError("db_path must be a Path")
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not isinstance(model, str):
        raise TypeError("model must be a string")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}. Run ingest.py first.")
        sys.exit(1)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model
    )

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
    except ValueError:
        print(f"Error: Collection '{collection_name}' not found. Run ingest.py first.")
        sys.exit(1)


def search(collection, query: str, n_results: int, threshold: float = None, source_filter: str = None, sort_latest: bool = False) -> list[dict]:
    """Search the vector database."""
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(n_results, int) or n_results <= 0:
        raise ValueError("n_results must be a positive integer")

    # Get more results if we're filtering or sorting
    fetch_n = n_results * 5 if (threshold or source_filter or sort_latest) else n_results

    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"]
    )

    matches = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i] if results.get("distances") else None
        metadata = results["metadatas"][0][i]

        # Apply threshold filter
        if threshold and distance and distance > threshold:
            continue

        # Apply source filter
        if source_filter:
            source = metadata.get("source", "")
            if not fnmatch.fnmatch(source.lower(), source_filter.lower()):
                continue

        matches.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": metadata,
            "distance": distance
        })

    # Sort by latest timestamp if requested
    if sort_latest:
        matches.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)

    # Limit results after sorting
    return matches[:n_results]


def show_stats(collection):
    """Show collection statistics."""
    count = collection.count()
    print(f"Collection: {collection.name}")
    print(f"Total chunks: {count}")

    if count > 0:
        results = collection.get(include=["metadatas"])
        sources = {}
        for m in results["metadatas"]:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        print(f"Unique sources: {len(sources)}")
        print(f"Avg chunks per source: {count / len(sources):.1f}")


def list_sources(collection):
    """List all unique source files."""
    results = collection.get(include=["metadatas"])
    sources = sorted(set(m.get("source", "unknown") for m in results["metadatas"]))

    print(f"Sources ({len(sources)} files):\n")
    for src in sources:
        print(f"  {src}")


def print_results(query: str, matches: list[dict], args):
    """Print search results."""
    if args.json:
        output = {
            "query": query,
            "count": len(matches),
            "results": matches
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    if not matches:
        print("No results found.")
        return

    for i, match in enumerate(matches, 1):
        source = match["metadata"].get("source", "unknown")
        header = match["metadata"].get("header", "unknown")
        distance = match.get("distance")
        timestamp = match["metadata"].get("timestamp", 0)

        dist_str = f" (dist: {distance:.4f})" if distance else ""
        time_str = ""
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            time_str = f" [{dt.strftime('%Y-%m-%d')}]"

        print(f"[{i}] {source} > {header}{time_str}{dist_str}")
        print("-" * 40)

        if not args.metadata_only:
            content = match["content"]
            truncate_len = 0 if args.no_truncate else args.truncate
            if truncate_len and len(content) > truncate_len:
                content = content[:truncate_len] + "..."
            print(content)

        print()


def interactive_mode(collection, args):
    """Run interactive search loop."""
    print("Brain Docs RAG - Interactive Mode")
    print("Commands: 'quit', 'stats', 'sources', 'help'")
    print(f"Options: -n {args.num_results}, threshold: {args.threshold or 'none'}\n")

    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ("quit", "exit", "q"):
            print("Bye!")
            break
        elif cmd == "stats":
            show_stats(collection)
            continue
        elif cmd == "sources":
            list_sources(collection)
            continue
        elif cmd == "help":
            print("  quit/exit/q  - Exit")
            print("  stats        - Show collection stats")
            print("  sources      - List source files")
            print("  <query>      - Search for query")
            continue

        matches = search(collection, query, args.num_results, args.threshold, args.source, args.latest)
        print_results(query, matches, args)


def main():
    args = parse_args()

    # Handle no-query modes
    if args.stats or args.list_sources or args.interactive:
        collection = get_collection(args.db_path, args.collection, args.model)

        if args.stats:
            show_stats(collection)
        elif args.list_sources:
            list_sources(collection)
        elif args.interactive:
            interactive_mode(collection, args)
        return

    # Require query for normal search
    if not args.query:
        print("Usage: python query.py \"your search query\"")
        print("       python query.py --interactive")
        print("       python query.py --stats")
        print("       python query.py --help")
        sys.exit(1)

    collection = get_collection(args.db_path, args.collection, args.model)
    matches = search(collection, args.query, args.num_results, args.threshold, args.source, args.latest)
    print_results(args.query, matches, args)


if __name__ == "__main__":
    main()
