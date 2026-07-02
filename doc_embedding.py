from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who is ms dhoni"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=2,
)

doc_embed = embeddings.embed_documents(documents)
query_embed = embeddings.embed_query(query)

similarity = cosine_similarity([query_embed], doc_embed)[0]

index, score = sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

