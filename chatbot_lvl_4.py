#conversation memory
from groq import Groq
import os

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
     



































