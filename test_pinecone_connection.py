from pinecone import Pinecone
import os

# Проверка подключения к Pinecone
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

print("API Key:", api_key)
print("Index Name:", index_name)

try:
    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)
    print("Successfully connected to Pinecone index.")
except Exception as e:
    print("Error connecting to Pinecone:", e)

# Обновление размерности вектора для соответствия индексу
vector = [0.1] * 512

try:
    results = index.query(vector=vector, top_k=10, include_metadata=True)
    print("Query Results:", results)
except Exception as e:
    print("Error during query:", e)