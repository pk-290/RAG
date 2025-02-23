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
from weaviate.classes.query import MetadataQuery


load_dotenv()
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


def create_rag_pipeline_google_embeddings(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Design a RAG pipeline that takes a PDF, loads, chunks, creates embeddings using Google Vertex AI,
    stores it in Weaviate, and retrieves documents based on a query.

    Args:
        pdf_path (str): Path to the PDF document.
        weaviate_url (str): URL of the Weaviate instance.
        weaviate_api_key (str): API key for Weaviate.
        chunk_size (int, optional): Size of text chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between text chunks. Defaults to 200.

    Returns:
        WeaviateVectorStore: Weaviate vector store object.
    """

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None  # Or raise the exception, depending on how you want to handle PDF loading failures

    # 2. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings - Using Google Vertex AI Embeddings
    try:
        embeddings = VertexAIEmbeddings(model="text-embedding-004")
    except Exception as e:
        print(f"Error initializing VertexAIEmbeddings: {e}")
        return None # Or raise the exception

    # 3. Create Embeddings - Using Google Vertex AI Embeddings
    # embeddings = VertexAIEmbeddings() # Removed model="text-embedding-004" - not a valid parameter

    # 4. Initialize Weaviate Client
    weaviate_cluster_url = "https://xek6qbd0s82eogfrldmcg.c0.asia-southeast1.gcp.weaviate.cloud"
    weaviate_api_key_val = "ANfSLS5yddxRHfKUdIbpM4DEaT6QniTNEb1y"

    auth_cred = Auth.api_key(weaviate_api_key_val)
    client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_cluster_url,
                auth_credentials=auth_cred)

    # 5. Store in Weaviate Vector Store
    vectorstore = WeaviateVectorStore.from_documents( # Use WeaviateVectorStore.from_documents
        texts,
        embeddings,
        client=client,
        index_name="RAG", # Choose an index name
        text_key="text",
    )

    return vectorstore


def retreive_document(query):
    weaviate_cluster_url = "https://xek6qbd0s82eogfrldmcg.c0.asia-southeast1.gcp.weaviate.cloud"
    weaviate_api_key_val = "ANfSLS5yddxRHfKUdIbpM4DEaT6QniTNEb1y"

    auth_cred = Auth.api_key(weaviate_api_key_val)
    client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_cluster_url,
                auth_credentials=auth_cred)
    
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    query_vector = embeddings.embed_query(query)
    collection = client.collections.get("RAG")
    response = collection.query.near_vector(
    near_vector=query_vector, # your query vector goes here
    limit=2,
    return_metadata=MetadataQuery(distance=True)
    )
    return response



if __name__ == '__main__':

    # vector_store_google = create_rag_pipeline_google_embeddings(
    #     pdf_path="RAG/deepseek scaling open source.pdf"
    # )

    # print("PDF document loaded, chunked, embedded with Google Embeddings, and stored in Weaviate.")

    # query_text = "deepseek model"
    # retrieved_documents_google = retrieve_documents_from_weaviate(vector_store_google, query_text)

    # print("\n--- Retrieved Documents (Google Embeddings) ---")
    # for doc in retrieved_documents_google:
    #     print(f"Page Content: {doc.page_content[:200]}...")
    #     print(f"Metadata: {doc.metadata}")
    #     print("-" * 50)

    query_text = "deepseek model"
    retrieved_documents = retreive_document(query_text)
    print("\n--- Retrieved Documents (Google Embeddings) ---")
    for o in retrieved_documents.objects:
        print(o.properties)
        print(o.metadata.distance)


    vector_store_google._client.close() # type:ignore
    print("Weaviate client connection closed.")






