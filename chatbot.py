from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

class ChatBot:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=("You are an Indian lawyer. Here is the conversation history:\n"
                      "{chat_history}\n"
                      "Now, answer the following question as an Indian lawyer would:\n"
                      "{question}")
        )
        self.llama = Ollama(model="llama3.1:8b-instruct-q3_K_M")
        self.chain = LLMChain(llm=self.llama, prompt=self.prompt_template)
        self.chat_history = []

    def chat(self, question):
        # Construct the history string
        history = "\n".join([f"Q: {entry['question']}\nA: {entry['response']}" for entry in self.chat_history])
        # Run the chain with the history and new question
        response = self.chain.run({"chat_history": history, "question": question})
        # Append the new interaction to the history
        self.chat_history.append({"question": question, "response": response})
        return response

    def show_history(self):
        for i, entry in enumerate(self.chat_history, 1):
            print(f"Q{i}: {entry['question']}")
            print(f"A{i}: {entry['response']}")
            print("-" * 50)

if __name__ == "__main__":
    bot = ChatBot()
    
    while True:
        user_input = input("Ask a legal question (or type 'exit' to quit, 'history' to view chat history): ").strip().lower()
        if user_input == "exit":
            break
        elif user_input == "history":
            bot.show_history()
        else:
            response = bot.chat(user_input)
            print(response)
