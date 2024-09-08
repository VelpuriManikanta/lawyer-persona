from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Define a simple mock retriever (Replace with actual retriever setup)
class SimpleRetriever:
    def retrieve(self, query):
        # This should be replaced with actual retrieval logic
        return f"Retrieved information for query: {query}"

class ChatBot:
    def __init__(self):
        # Initialize the retriever
        self.retriever = SimpleRetriever()
        
        # Define the prompt template with retrieval context
        self.prompt_template = PromptTemplate(
            input_variables=["chat_history", "retrieval", "question"],
            template=("You are an Indian lawyer. Here is the conversation history:\n"
                      "{chat_history}\n"
                      "Relevant information:\n"
                      "{retrieval}\n"
                      "Now, answer the following question as an Indian lawyer would:\n"
                      "{question}")
        )
        
        # Initialize the LLM with Ollama
        self.llama = Ollama(model="llama3.1:8b-instruct-q3_K_M")
        self.chain = LLMChain(llm=self.llama, prompt=self.prompt_template)
        self.chat_history = []

    def chat(self, question):
        # Construct the history string
        history = "\n".join([f"Q: {entry['question']}\nA: {entry['response']}" for entry in self.chat_history])
        
        # Retrieve relevant information
        retrieval = self.retriever.retrieve(question)
        
        # Run the chain with the history, retrieval information, and new question
        response = self.chain.run({"chat_history": history, "retrieval": retrieval, "question": question})
        
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
