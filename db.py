import weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Weaviate

# 1. Load the PDF document
pdf_path = "path/to/your/document.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Chunk the loaded document(s)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Initialize Google Vertex AI Embeddings
# Ensure your Google Cloud credentials are set in the environment.
embeddings = VertexAIEmbeddings()

# 4. Connect to Weaviate and set up the vector store
weaviate_url = "https://xek6qbd0s82eogfrldmcg.c0.asia-southeast1.gcp.weaviate.cloud"  # Update if your Weaviate instance is hosted elsewhere
client = weaviate.Client(weaviate_url)

# Define the schema for your index
index_name = "Document"
text_key = "text"

schema = {
    "classes": [{
        "class": index_name,
        "description": "A class to hold PDF document chunks",
        "vectorizer": "none",  # We're using external embeddings
        "properties": [
            {
                "name": text_key,
                "dataType": ["text"]
            }
        ]
    }]
}

# Create the schema if it does not exist.
if not client.schema.contains(schema):
    client.schema.create(schema)

# Initialize the Weaviate vector store using Vertex AI embeddings.
vectorstore = Weaviate(
    client,
    index_name,
    text_key,
    embedding_function=embeddings.embed_query  # Note: this function converts text into vectors
)

# Add the document chunks to the vector store.
vectorstore.add_documents(docs)

# 5. Retrieve documents based on a query.
query = "What is the main topic of the document?"  # Replace with your desired query
results = vectorstore.similarity_search(query)

print("Retrieved documents:")
for doc in results:
    print(doc.page_content)
