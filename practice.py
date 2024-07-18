from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Add documents to the vectorstore
documents = [
    Document(page_content="LangChain is a framework for building applications with LLMs."),
    Document(page_content="OpenAI provides powerful language models."),
    Document(page_content="Chroma is used for efficient similarity search.")
]
db = Chroma.from_documents(documents, embeddings, persist_directory="chroma")

# Retrieve documents from the vectorstore
query = "What is OpenAI?"
results = db.similarity_search(query, k=3)
print(results[0])
