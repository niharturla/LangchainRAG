from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model  = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

    
while True:
    print("\n\n..............................")
    question = input("Ask your question. (q to Quit)")
    print("\n\n")
    if question == "q":
        break

    res = chain.invoke({"reviews": [], "question": question})
    print(res)