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
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


class RAGPipeline:
    """
    Design a RAG pipeline that takes a PDF, loads, chunks, creates embeddings using Google Vertex AI,
    stores it in Weaviate, and retrieves documents based on a query.
    """

    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200,
                 weaviate_cluster_url=None, weaviate_api_key_val=None,
                 embedding_model="text-embedding-004", index_name="RAG", text_key="text"):
        """
        Initializes the RAGPipeline with configurations.

        Args:
            pdf_path (str): Path to the PDF document.
            chunk_size (int, optional): Size of text chunks. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between text chunks. Defaults to 200.
            weaviate_url (str): URL of the Weaviate instance.
            weaviate_api_key (str): API key for Weaviate.
            embedding_model (str, optional): Model for VertexAIEmbeddings. Defaults to "text-embedding-004".
            index_name (str, optional): Name of the Weaviate index. Defaults to "RAG".
            text_key (str, optional): Text key in Weaviate schema. Defaults to "text".
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.weaviate_cluster_url = weaviate_cluster_url if weaviate_cluster_url else "https://xek6qbd0s82eogfrldmcg.c0.asia-southeast1.gcp.weaviate.cloud" # Provide default or take from env
        self.weaviate_api_key_val = weaviate_api_key_val if weaviate_api_key_val else "ANfSLS5yddxRHfKUdIbpM4DEaT6QniTNEb1y" # Provide default or take from env
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.text_key = text_key
        self.vectorstore = None  # Initialize vectorstore as None
        self.client = None      # Initialize weaviate client as None

    def _load_and_chunk_pdf(self):
        """Loads PDF and chunks documents."""
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts

    def _create_embeddings(self):
        """Creates embeddings using Google Vertex AI Embeddings."""
        try:
            embeddings = VertexAIEmbeddings(model=self.embedding_model)
            return embeddings
        except Exception as e:
            print(f"Error initializing VertexAIEmbeddings: {e}")
            return None

    def _initialize_weaviate_client(self):
        """Initializes Weaviate client."""
        try:
            auth_cred = Auth.api_key(self.weaviate_api_key_val)
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_cluster_url,
                auth_credentials=auth_cred)
            self.client = client # Store client in instance
            return client
        except Exception as e:
            print(f"Error initializing Weaviate client: {e}")
            return None

    def _store_in_weaviate(self, texts, embeddings, client):
        """Stores documents and embeddings in Weaviate."""
        try:
            vectorstore = WeaviateVectorStore.from_documents(
                texts,
                embeddings,
                client=client,
                index_name=self.index_name,
                text_key=self.text_key,
            )
            return vectorstore
        except Exception as e:
            print(f"Error storing in Weaviate: {e}")
            return None

    def create_vector_store(self):
        """
        Orchestrates the RAG pipeline to create and store the vector store.

        Returns:
            WeaviateVectorStore: Weaviate vector store object, or None if pipeline fails.
        """
        texts = self._load_and_chunk_pdf()
        if texts is None:
            return None
        embeddings = self._create_embeddings()
        if embeddings is None:
            return None
        client = self._initialize_weaviate_client()
        if client is None:
            return None
        vectorstore = self._store_in_weaviate(texts, embeddings, client)
        self.vectorstore = vectorstore # Store vectorstore in instance
        return vectorstore

    def retrieve_documents(self, query, search_kwargs={"k": 3}):
        """
        Retrieves documents from Weaviate vector store based on a given query.

        Args:
            vectorstore (WeaviateVectorStore): Weaviate vector store object.
            query (str): The query string.
            search_kwargs (dict, optional): Keyword arguments for similarity search. Defaults to {"k": 3}.

        Returns:
            list: List of retrieved documents.
        """
        if self.vectorstore is None:
            print("Vector store is not initialized. Please call create_vector_store() first.")
            return None
        retrieved_docs = self.vectorstore.similarity_search(query, **search_kwargs)
        return retrieved_docs

    def close_weaviate_client(self):
        """Closes the Weaviate client connection."""
        if self.client: # Check if client is initialized
            self.client.close()
            print("Weaviate client connection closed.")


if __name__ == '__main__':

    rag_pipeline_google = RAGPipeline(
        pdf_path="RAG/deepseek scaling open source.pdf"
    )

    vector_store_google = rag_pipeline_google.create_vector_store()

    if vector_store_google:
        print("PDF document loaded, chunked, embedded with Google Embeddings, and stored in Weaviate.")

        query_text = "deepseek model"
        retrieved_documents_google = rag_pipeline_google.retrieve_documents(query_text)

        if retrieved_documents_google:
            print("\n--- Retrieved Documents (Google Embeddings) ---")
            for doc in retrieved_documents_google:
                print(f"Page Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
                print("-" * 50)

        rag_pipeline_google.close_weaviate_client()
    else:
        print("Failed to create vector store. Please check for errors.")