import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from components import *
#insert your actual Google API key Generated from the Google AI studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyB4bxdw9xqLQypyxW8D7RiFT-4MPHx0g24"
context = ['context\Shell-annual-report-2022.pdf','context\Tesla-annual-report-2022.pdf']
llm_model = "gemini-pro"
embed_model = "models/embedding-001"
print("Loading Context....")
docs = document_loader(doc_path=context[0])
chunked_context = splitter(docs)
for idx, con in enumerate(context):
    temp_doc = document_loader(doc_path=con)
    temp_chunked = splitter(temp_doc)
    temp_doc_store = vector_store(temp_chunked, embed_model=embed_model)
    if idx == 0:
        doc_stores = temp_doc_store
    else:
        doc_stores.merge_from(temp_doc_store)
    print(temp_doc[-1])
    print("Still working on it....")

retriever_system = doc_stores.as_retriever()

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
            response = retrieval_chain_with_souce.invoke(query)
            sources = []
            #for doc in response['context']:
            #    sources.append(doc.metadata['source'] + ", " + doc.metadata['page'])
            print(response['answer'])
            #print(sources)
            control = input("If you are done type e else type c: ")
            if control == "e":
                break
            elif control == "c":
                pass
            else:
                control = input("If you are done type e else type c: ")
            

