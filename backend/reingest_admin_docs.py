import asyncio
import sys
import hashlib

sys.stdout.reconfigure(encoding="utf-8")

from app.database.connection import db_manager
from app.ai.rag.document_loader import DocumentLoader
from app.ai.rag.text_splitter import TextSplitter
from app.ai.embeddings.local_provider import LocalEmbeddingProvider
from app.ai.vectorstore.faiss_store import FAISSVectorStore
from app.ai.vectorstore.base import VectorDocument
from app.core.config import settings


async def reingest():
    await db_manager.connect()
    db = db_manager.db

    # Get the 4 admin documents
    cursor = db["documents"].find({})
    docs = []
    async for doc in cursor:
        docs.append(doc)

    print("Found", len(docs), "admin documents")
    for doc in docs:
        print(" ", doc["document_id"], "-", doc["original_filename"])

    # Load embeddings provider
    embedding_provider = LocalEmbeddingProvider()
    dimension = embedding_provider.dimension

    # Create new FAISS store
    store = FAISSVectorStore(dimension=dimension, path=settings.VECTOR_STORE_PATH)
    store._create_new_index()

    # Process each document
    all_vector_docs = []
    for doc in docs:
        storage_path = settings.DOCUMENT_STORAGE_PATH + "/" + doc["storage_filename"]
        print("Loading:", storage_path)

        # Load and split document
        loader = DocumentLoader()
        pages = loader.load_document(doc["storage_filename"])
        content = "\n".join([p["content"] for p in pages])
        print("  Content length:", len(content))

        splitter = TextSplitter(chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP)
        text_chunks = splitter.split_text(content)
        chunks = [{"content": tc.content} for tc in text_chunks]
        print("  Chunks:", len(chunks))

        # Generate embeddings
        texts = [c["content"] for c in chunks]
        embeddings = embedding_provider.embed_texts(texts)
        print("  Embeddings generated:", len(embeddings))

        # Create vector documents
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5((doc["document_id"] + str(i)).encode()).hexdigest()[:32]
            vdoc = VectorDocument(
                id=doc["document_id"] + "_chunk_" + str(i),
                content=chunk["content"],
                embedding=embedding,
                metadata={
                    "source": doc["original_filename"],
                    "document_id": doc["document_id"],
                    "chunk_index": i,
                    "chunk_id": chunk_id
                }
            )
            all_vector_docs.append(vdoc)

        # Update chunk count in MongoDB (optional)
        await db["documents"].update_one(
            {"document_id": doc["document_id"]},
            {"$set": {"chunk_count": len(chunks)}}
        )

    # Add all to FAISS
    store.add_documents(all_vector_docs)
    store.persist()

    print("Total vectors in FAISS:", store._index.ntotal)


if __name__ == "__main__":
    asyncio.run(reingest())