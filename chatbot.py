from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

class ChatBot:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["question"],
            template="You are a lawyer. Answer the question as a lawyer would: {question}"
        )
        self.llama = Ollama(model="llama3.1:8b-instruct-q3_K_M")
        self.chain = LLMChain(llm=self.llama, prompt=self.prompt_template)

    def chat(self, question):
        response = self.chain.run({"question": question})
        return response

if __name__ == "__main__":
    bot = ChatBot()
    user_input = input("Ask a legal question: ")
    response = bot.chat(user_input)
    print(response)
