import warnings
from cryptography.utils import CryptographyDeprecationWarning
from langchain_google_genai import GoogleGenerativeAIEmbeddings # While imported, it's not used in the current code, consider removing if not needed
from langchain_text_splitters import CharacterTextSplitter # While imported, it's not used in the current code, consider removing if not needed
from langchain_community.document_loaders import PyPDFLoader
from weaviate.classes.init import Auth # While imported, it's not used in the corrected code, can be removed
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
import weaviate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from weaviate.classes.query import MetadataQuery
from RAG.connect_weaviate import get_weaviate_client
from dotenv import load_dotenv


load_dotenv()
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


def ingest_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None  # Or raise the exception, depending on how you want to handle PDF loading failures
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    client = get_weaviate_client()
    vectorstore = WeaviateVectorStore.from_documents(
        texts,
        embeddings,
        client=client,
        index_name="RAG", 
        text_key="text",
    )
    return vectorstore


if __name__ == '__main__':
    vector_store_google = ingest_document(pdf_path="RAG/when scaling meet llmm finetuning.pdf")
    print("PDF document loaded, chunked, embedded with Google Embeddings, and stored in Weaviate.")



