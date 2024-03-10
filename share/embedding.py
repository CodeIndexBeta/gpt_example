import pandas
import glob

from OpenAIClient import CreateClient

from llama_index.core import  SimpleDirectoryReader, GPTListIndex, PromptHelper

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

openAIClient = CreateClient()
path_to_training_data = "openAI_gpt/training_data/*"

def embedTrainingData():

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") 

    gfiles = glob.glob(path_to_training_data) 

    for g1 in range(len(gfiles)):
        f = open(f"embs{g1}.csv", "w") 
        f.write("combined") 
        f.close()

        content = ""

        with open(f"{gfiles[g1]}", 'r') as file: 
            content += file.read()
            content +=  "\n\n"


        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
        texts = text_splitter.split_text(content) 

        def get_embedding(text, model="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            embeddedResult = openAIClient.embeddings.create(input = [text], model=model).data[0].embedding
            print('embeddedResult')
            print(embeddedResult)
            return embeddedResult

        df = pandas.read_csv(f"embs{g1}.csv") 
        df["combined"] = texts 

        for i4 in range(len(df["combined"])):
            df["combined"][i4] =  '""'  + df["combined"][i4].replace("\n", "") +  '""'  

        df.to_csv(f"embs{g1}.csv") 

        df["embedding"] = df.combined.apply(lambda  x: get_embedding(x)) 
        df.to_csv(f"embs{g1}.csv", index=False) 
        df = pandas.read_csv(f"embs{g1}.csv") 
        embs = []

        for r1 in range(len(df.embedding)): 
            e1 = df.embedding[r1].split(",")
            for ei2 in range(len(e1)):
                e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
            embs.append(e1)

        df["embedding"] = embs 

        df.to_csv(f"embs{g1}.csv", index=False) 
