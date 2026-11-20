import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.document_loader import DocumentLoader
from app.ai.rag.text_splitter import TextSplitter
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.vectorstore.base import VectorDocument
import uuid

async def add_diabetes_document():
    await db_manager.connect()
    
    # Load the document
    loader = DocumentLoader()
    pages = loader.load_document("diabetes_education.txt")
    print(f"Loaded {len(pages)} page(s) from diabetes_education.txt")
    
    # Split into chunks
    splitter = TextSplitter()
    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page['content'], page['metadata'])
        all_chunks.extend(chunks)
    print(f"Split into {len(all_chunks)} chunks")
    
    # Generate embeddings
    embedding_provider = EmbeddingFactory.get_instance()
    texts = [chunk.content for chunk in all_chunks]
    embeddings = embedding_provider.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create VectorDocuments
    vector_docs = []
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        doc_id = f"doc_diabetes_education_chunk_{i}"
        vector_docs.append(VectorDocument(
            id=doc_id,
            content=chunk.content,
            embedding=embedding,
            metadata=chunk.metadata
        ))
    
    # Add to vector store
    store = VectorStoreFactory.get_instance()
    store.add_documents(vector_docs)
    store.persist()
    print(f"Added {len(vector_docs)} chunks to vector store")
    print(f"New total vectors: {store._index.ntotal}")

asyncio.run(add_diabetes_document())