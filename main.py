import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from components import *
#insert your actual Google API key Generated from the Google AI studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyB4bxdw9xqLQypyxW8D7RiFT-4MPHx0g24"
context = "context\shell-annual-report-2022.pdf"
llm_model = "gemini-pro"
embed_model = "models/embedding-001"

docs = document_loader(doc_path=context)
chunked_context = splitter(docs)
retriever_system = retriever(chunked_context, embed_model=embed_model)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompter()
    | LLM()
    | StrOutputParser()
)

retrieval_chain_with_souce = RunnableParallel(
    {"context": retriever_system, "query": RunnablePassthrough()}
).assign(answer=retrieval_chain)

if __name__ == "__main__":
    print ("Hi, I can read Tesla and Shell performance report of 2022")
    print("You can always press Ctr + C to exit")
    control = "c"
    while True:
        if control == "c":
            query = input("Ask you question here:\n")
            answer = retrieval_chain_with_souce.invoke(query)
            print(answer)
            control = input("If you are done type e else type c: ")
            if control == "e":
                break

