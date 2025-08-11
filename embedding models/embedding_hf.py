from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query("Capital of India is New Delhi")
print(str(result))