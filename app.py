import gradio as gr
from chatbot_backend import ask_chatbot

def chat_interface(user_message, history):
    # Call backend chatbot
    bot_reply = ask_chatbot(user_message)
    return bot_reply

# Create Gradio Chat UI
demo = gr.ChatInterface(
    fn=chat_interface,
    title="College Information Chatbot",
    description="Ask questions about the college. Powered by RAG + Ollama.",
)

# Launch the app
demo.launch()
