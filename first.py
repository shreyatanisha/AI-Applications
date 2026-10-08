# STEP 1-install dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq    #it helps to interact with your LLM model
from langchain.chains import create_retrieval_chain   # it helps create a retrieval-augmented generation (RAG) pipeline that combines document retrieval and language model generation.
from langchain.chains.combine_documents import create_stuff_documents_chain  #it helps to combine multiple documents into a single input for the language model. 
from langchain_core.prompts import ChatPromptTemplate  
from dotenv import load_dotenv
# ==========================================
# STEP 2: CONFIGURE API KEYS & PATHS
# ==========================================

load_dotenv()

# ==========================================
# STEP 3: LOAD THE PDF TEXT
# ==========================================
# Point directly to your PDF file path/point langchain to your file
loader = PyPDFLoader("data/CAREER_counsellor.pdf")

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
     model="openai/gpt-oss-20b",
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
prompt_template = ChatPromptTemplate.from_messages([   # here we define the prompt template that tells the model how to answer questions based on the retrieved chunks of text.
     ("system", system_instruction),                   # 
     ("human", "{input}"),
])


# ==========================================
# STEP 7: ASSEMBLE THE COMPLETE RAG CHAIN
# ==========================================
# 1. Turn our database into a retriever (fetches top 3 most relevant text blocks)
retriever = vector_database.as_retriever(search_kwargs={"k": 3})     #here     

# 2. Tell LangChain to bundle documents with the LLM prompt
document_chain = create_stuff_documents_chain(llm, prompt_template)  #here we create a document chain that combines the retrieved documents with the prompt template and sends them to the LLM for generating a response.

# 3. Create the final RAG chain
rag_pipeline = create_retrieval_chain(retriever, document_chain)  # here it combines the retriever and document chain into a single RAG pipeline that can take a user's question, retrieve relevant documents, and generate an answer using the LLM.

# ==========================================
# STEP 8: INTERACTIVE TERMINAL LOOP
# ==========================================
print("================================")
print("       MY AI CLIENT CHATBOT")
print("================================")
print("Type 'exit' to stop the chatbot.\n")

while True:
    user_question = input("Client: ").strip()

    if user_question.lower() == "exit":
        print("Bot: Thank you! Goodbye.")
        break

    result = rag_pipeline.invoke({"input": user_question})   # here we pass the user's question to the RAG pipeline, which retrieves relevant chunks from the vector database and generates a response using the Groq LLM. The result is a dictionary containing the answer.
    answer = result["answer"]

    print("Bot:", answer)
    print()
     


















































