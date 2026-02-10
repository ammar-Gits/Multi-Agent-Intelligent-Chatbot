# AI Chatbot with Persistent Memory

A conversational AI chatbot that maintains **persistent chat histories**, supports **multiple conversation threads**, and is built with **LangGraph**, **LangChain**, and **HuggingFace LLaMA**. The project uses **Streamlit** for the frontend and **SQLite** for storing chat states. Future enhancements include **RAG (Retrieval-Augmented Generation)** for knowledge-based responses.

## Features

- **Persistent Conversations:** Save and resume chats across sessions using SQLite.
- **Multiple Threads:** Start new chats or switch between previous conversations.
- **AI-Powered Responses:** Uses HuggingFace LLaMA-3.2-1B for natural language understanding.
- **Streamlit Frontend:** Interactive UI with real-time streaming of AI responses.
- **Future RAG Integration:** Planned support for knowledge-based, retrieval-augmented responses.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
Install dependencies:
pip install -r requirements.txt
Create a .env file with your HuggingFace API keys (if needed).
Usage
Run the backend:
python backend.py
Launch the Streamlit frontend:
streamlit run frontend.py
Open the Streamlit URL in your browser and start chatting.
Project Structure
ai-chatbot/
├── backend.py          # Chatbot backend and LangGraph implementation
├── frontend.py         # Streamlit frontend for interaction
├── chatbot.db          # SQLite database for storing chat states
├── requirements.txt    # Python dependencies
└── README.md
Future Enhancements
Integrate RAG (Retrieval-Augmented Generation) to provide responses based on external knowledge sources.
Add user authentication for multi-user support.
Improve UI/UX for a more interactive experience.
