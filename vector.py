from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("restaurants.csv")
embeddings = OllamaEmbeddings(model = "mxbai-embed-;arge")
db_loc = "./chrome_langchain_db"
add_documents = not os.path.exists(db_loc)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
vector_store = Chroma(
    collection_name = "restaurant_reviews",
    persist_directory=db_loc,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

print(df)