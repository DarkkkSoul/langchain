from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chat_history = []

while True:
    user_input = input("User: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    else:
        response = model.invoke(chat_history)
        answer = response.content[0]["text"]
        chat_history.append(answer)
        print("AI: ",answer)

print(chat_history)
