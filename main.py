import os
from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_graph_from_storage
from llama_index.core.response.pprint_utils import pprint_response
from sentence_transformers import SentenceTransformer

# Get the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

''' Set OpenAI API key (if required by other parts of LlamaIndex)
    os.environ["OPENAI_API_KEY"] = "sk-proj.."   '''

# Check if the key is available
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")

# Use the API key
os.environ["OPENAI_API_KEY"] = api_key
print(f"API Key: {api_key}")

# Define a custom embedding model using Hugging Face SentenceTransformer
class HuggingFaceEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self._model = SentenceTransformer(model_name)

    def _get_query_embedding(self, query: str):
        return self._model.encode(query)

    def _get_text_embedding(self, text: str):
        return self._model.encode(text)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

# Directory to store/load the index
index_cache = "./index_store"

# Check if the index exists; if not, create it
if not os.path.exists(index_cache):
    # Load documents from the local data directory
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"Loaded {len(documents)} documents.")

    # Initialize the custom Hugging Face embedding model
    hf_model = HuggingFaceEmbedding()

    # Create the index with the documents and custom embedding model
    index = VectorStoreIndex.from_documents(documents, embed_model=hf_model)

    # Save the index for future use
    index.storage_context.persist(persist_dir=index_cache)
    print("Index created and persisted.")
else:
    # Load the index from storage if it already exists
    storage_context = StorageContext.from_defaults(persist_dir=index_cache)
    index = load_graph_from_storage(storage_context)
    print("Index loaded from storage.")

# Create a query engine and execute a query
query_engine = index.as_query_engine(llm=None)  # Disable OpenAI LLM if not needed
response = query_engine.query("What is meditation?")

# Print the response with source information
pprint_response(response, show_source=True)
