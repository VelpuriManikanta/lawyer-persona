import tkinter as tk
import requests
import json

class ChatBot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Counseller")

        self.entry_field = tk.Entry(self.root, width=50)
        self.entry_field.grid(row=0, column=0, padx=10, pady=10)

        self.submit_button = tk.Button(self.root, text="Submit", command=self.check_input)
        self.submit_button.grid(row=0, column=1, padx=10, pady=10)

        self.response_label = tk.Label(self.root, text="", wraplength=400, justify="left")
        self.response_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def check_input(self):
        user_input = self.entry_field.get()
        if user_input:
            response = self.get_llama_response(user_input)
            self.response_label['text'] = response
        else:
            self.response_label['text'] = "Please enter a message"

    def get_llama_response(self, user_input):
        try:
            # Adding context to simulate a lawyer persona
            context = "You are a helpful and knowledgeable lawyer. Respond to the user's questions with professional and empathetic answers. Here is the question:"
            prompt = f"{context}\n{user_input}"

            response = requests.post(
                url="http://localhost:11434/api/chat",  # Update with the correct port if different
                json={
                    "model": "llama3.1:8b-instruct-q3_K_M",
                    "messages": [{"role": "system", "content": context}, {"role": "user", "content": user_input}]
                },
                stream=True  # Enable streaming mode
            )
            response.raise_for_status()

            # Read the response in chunks
            complete_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        data = json.loads(chunk.decode('utf-8'))
                        if 'message' in data:
                            complete_response += data['message'].get('content', '')
                    except json.JSONDecodeError:
                        pass

            return complete_response or "No response from model."

        except requests.exceptions.RequestException as e:
            return f"An error occurred: {str(e)}"

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()
