from langchain_google_genai import GoogleGenerativeAIEmbeddings
from weaviate.classes.query import MetadataQuery
from RAG.connect_weaviate import get_weaviate_client
from dotenv import load_dotenv

load_dotenv()

def retrieve_document(query):
    client = get_weaviate_client()
    collection = client.collections.get("RAG")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    query_vector = embeddings.embed_query(query)
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )
    return response