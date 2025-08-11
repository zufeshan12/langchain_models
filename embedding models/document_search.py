from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load environment variables
load_dotenv()

documents = ["GPT-5 is our flagship model for coding, reasoning, and agentic tasks across domains",
             "GPT-5 mini is a faster, more cost-efficient version of GPT-5. It's great for well-defined tasks and precise prompts.",
            "GPT-5 Nano is our fastest, cheapest version of GPT-5. It's great for summarization and classification tasks" ]

query = "tell me about GPT-5 Mini"

# define embedding model from OpenAI
model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)
# generate 300 dimensional embeddings
embedding_doc = model.embed_documents(documents)
embedding_query = model.embed_query(query)
# calculate cosine similarity for semantic comparison
similarity_scores = cosine_similarity([embedding_query],embedding_doc)[0]

print(f'Query : {query}')
print(f'Document matched : {documents[np.argmax(similarity_scores)]}')