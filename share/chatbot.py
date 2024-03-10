import gradio as gr
# from openai.embeddings_utils import cosine_similarity
# from sklearn.metrics.pairwise import cosine_similarity
import pandas
import numpy as np

from OpenAIClient import CreateClient

ips = []
ips_times = []

ips_ref = []
ips_times_ref = []

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

openAIClient = CreateClient()
llm = ChatOpenAI(temperature=0, openai_api_key="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", model_name="gpt-3.5-turbo")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    embeddedResult = openAIClient.embeddings.create(input = [text], model=model).data[0].embedding
    return embeddedResult

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def logic(question):
    
    df = pandas.read_csv(f"embs0.csv")

    embs = []
    for r1 in range(len(df.embedding)): 
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding(question) 
    print('product_embedding')
    print(product_embedding)
    print("\n\n\n\n\n")
    print("df.embedding")
    print(df.embedding)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding)) 
    df.to_csv("embs0.csv")

    df2 = df.sort_values("similarity", ascending=False) 
    df2.to_csv("embs0.csv")
    df2 = pandas.read_csv("embs0.csv")

    from langchain.docstore.document import Document

    comb = [df2["combined"][0]]
    docs = [Document(page_content=t) for t in comb]

    prompt_template = question + """

    {text}

    """ 

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT) 

    output = chain.run(docs) 

    return output

