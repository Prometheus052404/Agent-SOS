"""
Vector Store Module - ChromaDB-based semantic index for Xv6 agent.

Implements:
- ChromaDB persistent client for vector storage
- 500-word chunking with 100-word overlap
- Sentence-transformer embeddings (all-MiniLM-L6-v2)
- Lab relevance metadata filtering
- Code comment extraction from kernel/*.c
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("Sentence-transformers not available")


@dataclass
class Chunk:
    """Represents a text chunk for embedding."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Chunks text into overlapping segments."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        unit: str = 'words'
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit

    def chunk_text(
        self,
        text: str,
        base_id: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into overlapping chunks."""
        base_metadata = base_metadata or {}
        
        if self.unit == 'words':
            return self._chunk_by_words(text, base_id, base_metadata)
        else:
            return self._chunk_by_chars(text, base_id, base_metadata)

    def _chunk_by_words(
        self,
        text: str,
        base_id: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by word count."""
        words = text.split()
        chunks = []
        
        i = 0
        chunk_num = 0
        
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Chunk(
                id=f"{base_id}_chunk_{chunk_num}",
                text=chunk_text,
                metadata={
                    **base_metadata,
                    'chunk_index': chunk_num,
                    'word_start': i,
                    'word_end': end
                }
            ))
            
            chunk_num += 1
            i += self.chunk_size - self.chunk_overlap
        
        return chunks

    def _chunk_by_chars(
        self,
        text: str,
        base_id: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by character count."""
        chunks = []
        
        i = 0
        chunk_num = 0
        
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]
            
            chunks.append(Chunk(
                id=f"{base_id}_chunk_{chunk_num}",
                text=chunk_text,
                metadata={
                    **base_metadata,
                    'chunk_index': chunk_num,
                    'char_start': i,
                    'char_end': end
                }
            ))
            
            chunk_num += 1
            i += self.chunk_size - self.chunk_overlap
        
        return chunks


class CodeCommentExtractor:
    """Extracts comments from C source files."""

    def __init__(self):
        self.single_line_pattern = re.compile(r'//.*$', re.MULTILINE)
        self.multi_line_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)

    def extract_comments(
        self,
        filepath: str,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract all comments from a C file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return []
        
        comments = []
        lines = content.split('\n')
        
        # Extract single-line comments
        for i, line in enumerate(lines):
            match = self.single_line_pattern.search(line)
            if match:
                comment_text = match.group().lstrip('/ ')
                if len(comment_text) > 10:  # Skip very short comments
                    comments.append({
                        'text': comment_text,
                        'type': 'single_line',
                        'line': i + 1,
                        'file': filepath
                    })
        
        # Extract multi-line comments
        for match in self.multi_line_pattern.finditer(content):
            comment_text = match.group()
            # Clean the comment
            comment_text = comment_text[2:-2].strip()  # Remove /* and */
            comment_text = re.sub(r'^\s*\*\s?', '', comment_text, flags=re.MULTILINE)
            
            if len(comment_text) > 20:  # Skip very short comments
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                
                comments.append({
                    'text': comment_text,
                    'type': 'multi_line',
                    'line': line_num,
                    'file': filepath
                })
        
        return comments


class VectorStore:
    """
    ChromaDB-based vector store for textbook and code knowledge.
    """

    def __init__(
        self,
        persist_dir: str = ".xv6_agent/chroma_db",
        collection_name: str = "xv6_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self.chunker = TextChunker()
        self.comment_extractor = CodeCommentExtractor()
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.client = None
            self.collection = None
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

    def add_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add text to the vector store."""
        if not self.collection:
            logger.warning("ChromaDB not available, skipping add")
            return
        
        metadata = metadata or {}
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, doc_id, metadata)
        
        # Add chunks to collection
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        # Generate embeddings if available
        if self.embedding_model:
            embeddings = self.embedding_model.encode(texts).tolist()
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # Let ChromaDB handle embedding
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(chunks)} chunks from {doc_id}")

    def add_code_comments(self, filepath: str):
        """Extract and add comments from a C source file."""
        comments = self.comment_extractor.extract_comments(filepath)
        
        for comment in comments:
            doc_id = f"code_comment_{Path(filepath).stem}_{comment['line']}"
            
            self.add_text(
                text=comment['text'],
                doc_id=doc_id,
                metadata={
                    'source': 'code_comment',
                    'file': filepath,
                    'line': comment['line'],
                    'comment_type': comment['type']
                }
            )

    def add_textbook_content(
        self,
        text: str,
        source: str,
        chapter: Optional[int] = None,
        page: Optional[int] = None,
        topics: Optional[List[str]] = None,
        lab_relevance: Optional[List[str]] = None
    ):
        """Add textbook content with metadata."""
        doc_id = f"{source}_ch{chapter or 0}_p{page or 0}"
        
        metadata = {
            'source': source,
            'type': 'textbook'
        }
        
        if chapter:
            metadata['chapter'] = chapter
        if page:
            metadata['page'] = page
        if topics:
            metadata['topics'] = ','.join(topics)
        if lab_relevance:
            metadata['lab_relevance'] = ','.join(lab_relevance)
        
        self.add_text(text, doc_id, metadata)

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        lab_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            lab_filter: Filter by lab relevance
            source_filter: Filter by source type
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        if not self.collection:
            logger.warning("ChromaDB not available, returning empty results")
            return []
        
        # Build where clause
        where = {}
        if lab_filter:
            where['lab_relevance'] = {"$contains": lab_filter}
        if source_filter:
            where['source'] = source_filter
        
        # Generate query embedding
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query_text]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where if where else None
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where if where else None
            )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results and results['ids']:
            for i, chunk_id in enumerate(results['ids'][0]):
                score = 1.0  # Default score
                if results.get('distances') and results['distances'][0]:
                    # Convert distance to similarity (cosine)
                    score = 1 - results['distances'][0][i]
                
                if score >= min_similarity:
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        text=results['documents'][0][i] if results.get('documents') else "",
                        score=score,
                        metadata=results['metadatas'][0][i] if results.get('metadatas') else {}
                    ))
        
        return search_results

    def query_for_context(
        self,
        query_text: str,
        task_id: Optional[str] = None,
        max_tokens: int = 800
    ) -> str:
        """
        Query and format results for LLM context.
        
        Args:
            query_text: Query string
            task_id: Current task/lab ID for filtering
            max_tokens: Maximum tokens for results
            
        Returns:
            Formatted string of relevant chunks
        """
        results = self.query(
            query_text=query_text,
            n_results=3,
            lab_filter=task_id,
            min_similarity=0.3
        )
        
        if not results:
            return ""
        
        # Format results
        chunks = []
        total_length = 0
        max_chars = max_tokens * 4  # Rough estimate: 4 chars per token
        
        for result in results:
            if total_length + len(result.text) > max_chars:
                break
            
            source_info = result.metadata.get('source', 'unknown')
            if result.metadata.get('chapter'):
                source_info += f" Ch.{result.metadata['chapter']}"
            if result.metadata.get('page'):
                source_info += f" p.{result.metadata['page']}"
            
            chunk = f"[{source_info}] {result.text}"
            chunks.append(chunk)
            total_length += len(chunk)
        
        return '\n\n'.join(chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.collection:
            return {'status': 'unavailable'}
        
        count = self.collection.count()
        
        return {
            'status': 'available',
            'collection': self.collection_name,
            'chunk_count': count,
            'persist_dir': str(self.persist_dir)
        }

    def clear(self):
        """Clear all data from the collection."""
        if self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store cleared")


class FallbackVectorStore:
    """
    Simple keyword-based fallback when ChromaDB is not available.
    Uses TF-IDF-like scoring.
    """

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}

    def add_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add text to the store."""
        self.documents[doc_id] = {
            'text': text.lower(),
            'metadata': metadata or {}
        }

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        **kwargs
    ) -> List[SearchResult]:
        """Query using keyword matching."""
        query_words = set(query_text.lower().split())
        
        scores = []
        for doc_id, doc in self.documents.items():
            doc_words = set(doc['text'].split())
            overlap = len(query_words & doc_words)
            score = overlap / (len(query_words) + 0.1)
            scores.append((doc_id, score, doc))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score, doc in scores[:n_results]:
            if score > 0:
                results.append(SearchResult(
                    chunk_id=doc_id,
                    text=doc['text'][:500],  # Truncate
                    score=score,
                    metadata=doc['metadata']
                ))
        
        return results


def create_vector_store(
    persist_dir: str = ".xv6_agent/chroma_db"
) -> VectorStore:
    """Create a vector store instance."""
    return VectorStore(persist_dir=persist_dir)


if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.DEBUG)
    
    store = create_vector_store()
    
    # Add some test content
    store.add_textbook_content(
        text="Spinlocks are the simplest form of locks. They provide mutual exclusion by spinning in a loop until the lock becomes available.",
        source="xv6-book",
        chapter=4,
        page=23,
        topics=["locks", "spinlock"],
        lab_relevance=["lock_lab"]
    )
    
    store.add_textbook_content(
        text="The scheduler runs in a loop, finding a runnable process and switching to it. It holds the process table lock while scanning.",
        source="xv6-book",
        chapter=5,
        page=45,
        topics=["scheduler", "process"],
        lab_relevance=["thread_lab"]
    )
    
    # Query
    results = store.query("why does spinlock panic", n_results=2)
    print("\n=== Query Results ===")
    for r in results:
        print(f"Score: {r.score:.2f}")
        print(f"Text: {r.text[:100]}...")
        print()
    
    # Print stats
    stats = store.get_stats()
    print(f"\n=== Stats ===")
    print(stats)
