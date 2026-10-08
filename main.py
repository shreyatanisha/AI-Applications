# STEP 1-install dependencies
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# STEP 2: CONFIGURE API KEYS & PATHS
# ==========================================
api_key = os.environ["GROQ_API_KEY"]

# ==========================================
# STEP 3: LOAD THE PDF TEXT
# ==========================================# Point directly to your PDF file path/point langchain to your file
loader = PyPDFLoader("./data/CAREER_counsellor.pdf")

# Extract pages as individual document elements
raw_documents = loader.load()

print(f"Successfully loaded {len(raw_documents)} pages from the PDF.")

# ==========================================
# STEP 4: SPLIT THE PDF TEXT INTO SMALLER CHUNKS
# ==========================================
# We want chunks of 1,000 characters, overlapping by 200 characters 
# so sentences don't get awkwardly cut in half.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Chop up our extracted PDF text
split_docs = text_splitter.split_documents(raw_documents)

# ==========================================
# STEP 5: CONVERTING TEXT TO VECTORS AND SAVE TO DATABSE
# ==========================================

# Load the mathematical text translator (runs locally on your PC for free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a database on your PC using your chunks and embeddings
vector_database = Chroma.from_documents(split_docs, embeddings)

# ==========================================
# STEP 6: SET UP GROQ LLM & PROMPT
# ==========================================
# Initialize Groq's model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2  # Low temperature makes the AI strict and factual
)

# Set the rules for the AI
system_instruction = (
    "You are a helpful assistant. Use the provided context below to answer "
    "the user's question. If you don't know the answer based on the context, "
    "honestly say that you don't know. Do not make things up.\n\n"
    "Context:\n{context}"
)

# Put the prompt template together
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "{input}"),
])

# ==========================================
# STEP 7: ASSEMBLE THE COMPLETE RAG CHAIN
# ==========================================
# 1. Turn our database into a retriever (fetches top 3 most relevant text blocks)
retriever = vector_database.as_retriever(search_kwargs={"k": 3})

# 2. Tell LangChain to bundle documents with the LLM prompt
document_chain = create_stuff_documents_chain(llm, prompt_template)

# 3. Create the final RAG chain
rag_pipeline = create_retrieval_chain(retriever, document_chain)

# ==========================================
# STEP 8: INTERACTIVE TERMINAL LOOP
# ================================s==========
#conversation memory 
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# here we created a python list(conversation[]) where it will Store the conversation history 
conversation = [
    {
        "role": "system",
        "content": """
        You are a professional customer support chatbot.

        Your job is to help clients by answering their questions.

        Follow these rules:
        1. Be polite and professional.
        2. Keep answers simple and easy to understand.
        3. Give clear and useful answers.
        4. If you don't know something, say that you don't know.
        5. Do not make up information.
        """
    }
]

print("================================")
print("       MY AI CLIENT CHATBOT")
print("================================")
print("Type 'exit' to stop the chatbot.\n")

while True:

     #get question from client
     user_question = input("Client: ")

     #stop chatbot
     if user_question.lower() == "exit":
          print("Bot: Thank you! Goodbye.")
          break

      #Add user question to conversation
     conversation.append({
        "role": "user",
        "content": user_question 
      })

     #Send complete converastion to llm
     response = client.chat.completions.create(
          model= "openai/gpt-oss-20b",
          messages=conversation
     )

     #Get AI answer
     answer = response.choices[0].message.content

     #add ai answer to conversation
     conversation.append({
          "role": "assistant",
          "content": answer
    
     })

     #display answer
     print("Bot: ",answer)
     print()
     


















































