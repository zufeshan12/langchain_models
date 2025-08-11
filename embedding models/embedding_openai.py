from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)
text = "The capital of India is New Delhi"

result = embedding.embed_query(text=text)
print(str(result))

