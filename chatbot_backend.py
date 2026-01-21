from sentence_transformers import SentenceTransformer
import faiss
import pickle
import subprocess

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("college_embeddings.index")

# Load original college text
with open("college_embeddings.pkl", "rb") as f:
    college_text = pickle.load(f)

def ask_chatbot(user_question):
    # Convert question to embedding
    question_embedding = embed_model.encode([user_question])

    # Search in FAISS (top 3 matches)
    distances, indices = index.search(question_embedding, k=3)

    # Retrieve relevant context
    context = ""
    for idx in indices[0]:
        context += college_text[idx] + "\n"

    # Create prompt for Ollama
    prompt = f"""
Use the following college information to answer the question.

College Information:
{context}

Question:
{user_question}

Answer:
"""

    # Call Ollama (local model)
    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    answer = result.stdout.decode("utf-8", errors="ignore")
    return answer.strip()


# Test the backend directly
if __name__ == "__main__":
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        print("Bot:", ask_chatbot(q))
