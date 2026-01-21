from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("college_data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Better chunking
texts = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "college_embeddings.index")

with open("college_embeddings.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Embeddings and text saved successfully")
