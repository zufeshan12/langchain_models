from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(verbose=True) #load the api keys set as env variables

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b",
                          task="text-summarization")
                          
model = ChatHuggingFace(llm=llm)
text = "I witnessed some intense racism today on the streets of San Francisco with one of my Indian friends. A guy screamed at her, “go back to India, you’re messing up this country!” It’s wild to think where the US would be without the heroism of Indian doctors during the pandemic. Indians make up 1.6% of the US population but 8.5% of our doctors. While other people might’ve forgotten how they showed up for America during the pandemic, I have a deep appreciation for how you blessed America during those difficult times."
result = model.invoke(f"summarize the text. text={text}")
print(result.content)

