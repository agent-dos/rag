"""
Ingest documents into ChromaDB vector database.

Usage:
    python ingest.py                           # Default: ingest docs/brain
    python ingest.py --source ./my-docs        # Custom source
    python ingest.py --chunk-by fixed --chunk-size 500
    python ingest.py --dry-run                 # Preview without ingesting
    python ingest.py --clear                   # Clear collection first
    python ingest.py --verbose                 # Show detailed output
"""

import argparse
import re
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# Defaults
DEFAULT_SOURCE = Path(__file__).parent.parent / "docs" / "legacy" / "v2" / "brain"
DEFAULT_DB_PATH = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "brain_docs"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source directory with markdown files (default: {DEFAULT_SOURCE})"
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
        "--chunk-by",
        choices=["headers", "fixed"],
        default="headers",
        help="Chunking strategy: 'headers' (split by ##/###) or 'fixed' (fixed size)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for fixed chunking (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap between chunks for fixed chunking (default: {DEFAULT_OVERLAP})"
    )
    parser.add_argument(
        "--glob", "-g",
        type=str,
        default="*.md",
        help="Glob pattern for files (default: *.md)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before ingesting"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing collection (default: replace)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without actually doing it"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    return parser.parse_args()


def get_files(directory: Path, pattern: str) -> list[Path]:
    """Get files matching pattern from directory."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    return sorted(directory.glob(pattern))


def extract_timestamp(filename: str) -> int:
    """
    Extract timestamp from filename.
    Expected format: YYYYMMDD-HHMMSS-description.md
    Returns unix timestamp (int) or 0 if not found.
    """
    if not isinstance(filename, str):
        return 0

    # Match YYYYMMDD-HHMMSS pattern at start of filename
    match = re.match(r'^(\d{8})-(\d{6})', filename)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        try:
            dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            return int(dt.timestamp())
        except ValueError:
            pass

    # Try YYYYMMDD only
    match = re.match(r'^(\d{8})', filename)
    if match:
        date_part = match.group(1)
        try:
            dt = datetime.strptime(date_part, "%Y%m%d")
            return int(dt.timestamp())
        except ValueError:
            pass

    return 0


def chunk_by_headers(content: str, filename: str) -> list[dict]:
    """Split markdown content by headers (##, ###)."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    chunks = []
    pattern = r'(^#{2,3}\s+.+$)'
    parts = re.split(pattern, content, flags=re.MULTILINE)

    current_header = filename
    current_content = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r'^#{2,3}\s+', part):
            if current_content.strip():
                chunks.append({
                    "header": current_header,
                    "content": current_content.strip(),
                    "source": filename
                })
            current_header = re.sub(r'^#{2,3}\s+', '', part)
            current_content = ""
        else:
            current_content += part + "\n"

    if current_content.strip():
        chunks.append({
            "header": current_header,
            "content": current_content.strip(),
            "source": filename
        })

    if not chunks and content.strip():
        chunks.append({
            "header": filename,
            "content": content.strip(),
            "source": filename
        })

    return chunks


def chunk_by_fixed_size(content: str, filename: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split content into fixed-size chunks with overlap."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    chunks = []
    text = content.strip()

    if not text:
        return chunks

    start = 0
    chunk_num = 1

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at word boundary
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            if last_space > chunk_size // 2:
                chunk_text = chunk_text[:last_space]
                end = start + last_space

        chunks.append({
            "header": f"{filename} [chunk {chunk_num}]",
            "content": chunk_text.strip(),
            "source": filename
        })

        start = end - overlap
        chunk_num += 1

    return chunks


def ingest_documents(args):
    """Main ingestion function."""
    print(f"Source: {args.source}")
    print(f"Database: {args.db_path}")
    print(f"Collection: {args.collection}")
    print(f"Chunking: {args.chunk_by}" + (f" (size={args.chunk_size}, overlap={args.overlap})" if args.chunk_by == "fixed" else ""))
    print()

    # Get files
    files = get_files(args.source, args.glob)
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files")

    # Process files into chunks
    all_chunks = []
    for file_path in files:
        if args.verbose:
            print(f"  Processing: {file_path.name}")

        content = file_path.read_text(encoding="utf-8")

        if args.chunk_by == "headers":
            chunks = chunk_by_headers(content, file_path.name)
        else:
            chunks = chunk_by_fixed_size(content, file_path.name, args.chunk_size, args.overlap)

        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    if args.dry_run:
        print("\n[DRY RUN] Would ingest the following:")
        for i, chunk in enumerate(all_chunks[:10]):
            print(f"  [{i+1}] {chunk['source']} > {chunk['header'][:50]}...")
        if len(all_chunks) > 10:
            print(f"  ... and {len(all_chunks) - 10} more chunks")
        return

    # Setup ChromaDB
    print(f"\nLoading embedding model: {args.model}...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model
    )

    client = chromadb.PersistentClient(path=str(args.db_path))

    # Handle collection
    existing = [c.name for c in client.list_collections()]

    if args.collection in existing:
        if args.clear or not args.append:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection: {args.collection}")
            collection = client.create_collection(
                name=args.collection,
                embedding_function=embedding_fn
            )
        else:
            collection = client.get_collection(
                name=args.collection,
                embedding_function=embedding_fn
            )
            print(f"Appending to existing collection: {args.collection}")
    else:
        collection = client.create_collection(
            name=args.collection,
            embedding_function=embedding_fn
        )

    # Generate unique IDs for append mode
    if args.append and args.collection in existing:
        existing_count = collection.count()
        start_id = existing_count
    else:
        start_id = 0

    # Add chunks with timestamp metadata
    if all_chunks:
        metadatas = []
        for c in all_chunks:
            timestamp = extract_timestamp(c["source"])
            metadatas.append({
                "header": c["header"],
                "source": c["source"],
                "timestamp": timestamp
            })

        collection.add(
            ids=[f"chunk_{start_id + i}" for i in range(len(all_chunks))],
            documents=[c["content"] for c in all_chunks],
            metadatas=metadatas
        )
        print(f"Added {len(all_chunks)} chunks to collection '{args.collection}'")

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    ingest_documents(args)
