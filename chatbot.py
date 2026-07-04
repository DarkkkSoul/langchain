from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant")
]

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    else:
        chat_history.append(HumanMessage(content = user_input))
        response = model.invoke(chat_history)
        answer = response.content[0]["text"]
        chat_history.append(AIMessage(content=answer))
        print("AI: ",answer)

print(chat_history)
