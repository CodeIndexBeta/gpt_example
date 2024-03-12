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
# from langchain_openai import ChatOpenAI
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
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

openAIClient = CreateClient()
llm = ChatOpenAI(temperature=0, openai_api_key="", model_name="gpt-3.5-turbo")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    embeddedResult = openAIClient.embeddings.create(input = [text], model=model).data[0].embedding
    return embeddedResult

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def logic(question):
    
    df = pandas.read_csv(f"embs0.csv")

    embs = []
    for r1 in range(len(df.embedding)): # Changing the format of the embeddings into a list due to a parsing error
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding(question) # Creating an embedding for the question that's been asked
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding)) # Finds the relevance of each piece of data in context of the question
    df.to_csv("embs0.csv")

    df2 = df.sort_values("similarity", ascending=False) # Sorts the text chunks based on how relevant they are to finding the answer to the question
    df2.to_csv("embs0.csv")
    df2 = pandas.read_csv("embs0.csv")

    from langchain.docstore.document import Document

    comb = [df2["combined"][0]]
    docs = [Document(page_content=t) for t in comb] # Gets the most relevant text chunk

    # prompt_template = question + """{text}""" + "Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION. If the instruction is about placing a trade then return a response in JSON format and the object should contain 3 keys. 1st QUANTITY specified by user FROM token/product user want to trade TO token/product user wants to trade with."
    # print(prompt_template)
    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # prompt = PromptTemplate(template= question + """
    # Do not justify your answers and do not give information not mentioned in the provided embeddings.

    # If the instruction is about placing a trade then please note these points: Return the response in JSON format,
    # The JSON object should contains the following fields - TO_PRICE, TO, FROM_PRICE and FROM.
    # Here are the details that you have to look for mapping the correct data to each key,
    # Price of TO and FROM will be specified by the user and if only 1 price is provided then it is assumed that the price of
    # both i.e. TO_PRICE and FROM_PRICE will be same. The price field should be a number and can be float too so take care of the JSON object accordingly.
    # The the of token/product user wants to trade will be provided and should be mapped in TO and FROM fields accodingly.

    # If you feel some information is missing or the user is asking questions outside the scope of the provided embedding then you should
    # return a gracefull message for example - Provided information is not complete or I am not trained to answer such questions.

    # {text}
    # """, input_variables=["text"])

    # instruction_context = """
    # Do not justify your answers and do not give information not mentioned in the provided embeddings.
    # """
    # instruction_placing_order = """
    # If the instruction is about placing a trade then please note these points: Return the response in JSON format,
    # The JSON object should contains the following fields - TO_PRICE, TO, FROM_PRICE and FROM.
    # Here are the details that you have to look for mapping the correct data to each key,
    # Price of TO and FROM will be specified by the user and if only 1 price is provided then it is assumed that the price of
    # both i.e. TO_PRICE and FROM_PRICE will be same. The price field should be a number and can be float too so take care of the JSON object accordingly.
    # The the of token/product user wants to trade will be provided and should be mapped in TO and FROM fields accodingly.
    # """
    # instruction_error="""
    # If you feel some information is missing or the user is asking questions outside the scope of the provided embedding then you should
    # return a gracefull message for example - Provided information is not complete or I am not trained to answer such questions.
    # """
    # template = question + """ {instruction_context} {instruction_placing_order} {instruction_error} {text}"""
    # print("template")
    # print(template)
    # new_prompt = PromptTemplate(template=template, input_variables=["instruction_context", "instruction_placing_order", "instruction_error", "text"])
    # print("new_prompt")
    # print(new_prompt)

    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=new_prompt, document_variable_name=["instruction_context","text"]) # Preparing the LLM
    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, input_variables="text") # Preparing the LLM
    # output = chain.run(docs) # Formulating an answer (this is where the magic happens)


    # # Define prompt
    # prompt = PromptTemplate.from_template( question + """
    # Use the following pieces of context to answer the question at the end. If you 
    # don't know the answer, just say that you don't know, don't try to make up an 
    # answer.

    # =====================================================================================================================================
    # PERSONAL DETAILS
    # =====================================================================================================================================
    # Here are few details about you.
    # - You are Mettalex GPT.
    # - You have been developed by Mettalex.
    # - You were born on 11th March 2024
    # =====================================================================================================================================

    # =====================================================================================================================================
    # DETAILS ON PLACING A TRADE ONLY
    # =====================================================================================================================================
    # - Be sure if user is asking for a trade pair then only consider the details for DETAILS ON PLACING A TRADE ONLY else skip.
    # - Return the response in JSON format,
    # - The JSON object should contains the following fields - TO_PRICE, TO, FROM_PRICE and FROM.
    # - Price of TO and FROM will be specified by the user, if only one price is provided then use same price for both i.e. TO_PRICE and FROM_PRICE will be same.
    # - The price field should be a number and make sure you keep the number same as provided so take care of the JSON object accordingly.
    #   For example user can provide price as 1, 100, 0.1, 10.235, 1.00215, 0.002 and etc.
    # - The value of token/product user wants to trade will be provided and should be mapped in TO and FROM fields accodingly.
    # - The supported trade pairs strictly are FET, USDT and ATESTFET. If user is providing any other value for TO and FROM than this, the please return error message
    #   saying - Trade pair not supported.
    # - If the value of TO and FROM are other than FET, USDT and ATESTFET then please return error -Trade pair not supported.
    # =====================================================================================================================================


    
    # {context}
    
    # """)


    prompt = PromptTemplate.from_template( question + """
    Mettalex GPT, an AI developed by Mettalex, was born on March 11, 2024. 
    When it comes to placing trades, specific guidelines must be followed. 
    Requests for trade pairs should be responded to in JSON format, including fields such as TO_PRICE, TO, FROM_PRICE, and FROM. 
    Users can input prices for the tokens they wish to trade, with the option to specify the same price for both TO and FROM tokens. 
    The requested tokens for trading will be allocated accordingly to the TO and FROM fields. 
    It's crucial to note that only trade pairs involving FET, USDT, and ATESTFET are supported. 
    Any other values inputted for TO and FROM will result in an error message indicating that the trade pair is not supported.
    {context}
    """)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    output = stuff_chain.run(docs)

    return output

