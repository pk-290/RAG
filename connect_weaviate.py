import os
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

def get_weaviate_client():
    weaviate_cluster_url = os.getenv("WEAVIATE_CLUSTER_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_cluster_url:
        raise ValueError("Missing environment variable: WEAVIATE_CLUSTER_URL")
    if not weaviate_api_key:
        raise ValueError("Missing environment variable: WEAVIATE_API_KEY")
        
    auth_cred = Auth.api_key(weaviate_api_key)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_cluster_url,
        auth_credentials=auth_cred
    )
    return client



