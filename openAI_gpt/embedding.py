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

    # Setting your OpenAI model
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") 

    # Reading your document directory
    gfiles = glob.glob(path_to_training_data) 

    # Reading all the documents
    for g1 in range(len(gfiles)):
        # Creating a csv file for storing the embeddings for your ChatBot
        f = open(f"embs{g1}.csv", "w") 
        # Creating the 'combined' collumn
        f.write("combined") 
        f.close()

        content = ""

        # Storing the document contents
        with open(f"{gfiles[g1]}", 'r') as file: 
            content += file.read()
            content +=  "\n\n"


        # Splitting the document content into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
        texts = text_splitter.split_text(content) 

        # Defining the function that creates the embeddings needed for the Chatbot to function (It can't form answers from plain text)
        def get_embedding(text, model="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            embeddedResult = openAIClient.embeddings.create(input = [text], model=model).data[0].embedding
            # embeddedResult = openAIClient.embeddings.create(input = [text], model=model, dimensions=2).data[0].embedding
            print('embeddedResult')
            print(embeddedResult)
            return embeddedResult

        # Reading the empty csv file that you created earlier for storing the embeddings
        df = pandas.read_csv(f"embs{g1}.csv") 
        # Filling the 'combined' collumn with the chunks you created earlier
        df["combined"] = texts 

        # Adding triple quotes around the text chunks to prevent syntax errors caused by double quotes in the text
        for i4 in range(len(df["combined"])):
            df["combined"][i4] =  '""'  + df["combined"][i4].replace("\n", "") +  '""'  

        # Writing the data to the csv file
        df.to_csv(f"embs{g1}.csv") 

        # Adding and filling the 'embedding' collumn which contains the embeddings created from your text chunks
        df["embedding"] = df.combined.apply(lambda  x: get_embedding(x)) 
        # Writing the new 'embedding' collumn to the csv file
        df.to_csv(f"embs{g1}.csv", index=False) 
        # Reading the new csv file
        df = pandas.read_csv(f"embs{g1}.csv") 
        embs = []

        # Making the embeddings readable to the chatbot by turning them into lists
        for r1 in range(len(df.embedding)): 
            e1 = df.embedding[r1].split(",")
            for ei2 in range(len(e1)):
                e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
            embs.append(e1)

        # Updating the 'embedding' collumn
        df["embedding"] = embs 

        # Writing the final version of the csv file
        df.to_csv(f"embs{g1}.csv", index=False) 
