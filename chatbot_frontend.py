import streamlit as st
import uuid
from rag_chatbot_backend import (
    chatbot,
    get_all_threads,
    delete_thread,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata,
)
from langchain.schema import HumanMessage, BaseMessage, AIMessage, SystemMessage

# --------------------------
# Helper Functions
# --------------------------

def generate_thread_id():
    return str(uuid.uuid4())

def _one_line(text: str) -> str:
    return " ".join((text or "").split())

def _truncate(text: str, max_len: int = 40) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"

def get_conversation_title(thread_id: str, prefer: str = "first_user") -> str:
    """
    Create a human-friendly conversation title from stored messages.

    prefer:
      - "first_user": first non-empty user message
      - "last_user": last non-empty user message
    """
    conversation = load_conversation(thread_id)
    user_texts = [
        _one_line(m.get("content", ""))
        for m in conversation
        if m.get("role") == "user" and _one_line(m.get("content", ""))  # non-empty
    ]

    chosen = ""
    if user_texts:
        chosen = user_texts[0] if prefer == "first_user" else user_texts[-1]

    if not chosen:
        # Fallback: show a short stable identifier so empty threads are still selectable
        return f"New conversation ({thread_id[:8]})"

    return _truncate(chosen, max_len=48)

def clear_chat():
    """Start a new chat thread with no PDF; reset file uploader for a clean start."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    st.session_state['rag_pdf_upload_key'] = st.session_state.get('rag_pdf_upload_key', 0) + 1

def add_thread(thread_id):
    """
    Add or bump a thread in the list of threads so that
    the most recently active conversation appears at the top.
    """
    threads = st.session_state['chat_threads']
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)

def load_conversation(thread_id):
    """Load the message history for a given thread from chatbot state."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    conversation = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        conversation.append({'role': role, 'content': msg.content})
    return conversation

# --------------------------
# Session State Initialization
# --------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'rag_pdf_upload_key' not in st.session_state:
    st.session_state['rag_pdf_upload_key'] = 0

# --------------------------
# Sidebar UI
# --------------------------

st.sidebar.title("Multi Agent Chatbot")

if st.sidebar.button("New Chat"):
    clear_chat()
    st.rerun()

# PDF upload for RAG: answers in this thread use only the uploaded PDF when present
st.sidebar.subheader("Upload PDF here")
uploaded_pdf = st.sidebar.file_uploader(
    "Add a PDF to this conversation",
    type=["pdf"],
    key=f"rag_pdf_upload_{st.session_state['rag_pdf_upload_key']}",
)
thread_id = st.session_state["thread_id"]
if thread_has_document(thread_id):
    meta = thread_document_metadata(thread_id)
    st.sidebar.caption(f"📄 **{meta.get('filename', 'PDF')}** — RAG only for this chat")

st.sidebar.title("My Conversations")

# Always use the first user message as the conversation name
prefer_key = "first_user"

for thread_id in st.session_state['chat_threads']:
    cols = st.sidebar.columns([4, 1])
    label = get_conversation_title(thread_id, prefer=prefer_key)

    # Select conversation
    if cols[0].button(label, key=f"thread-{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)

    # Delete conversation
    if cols[1].button("Del", key=f"delete-{thread_id}"):
        delete_thread(thread_id)
        # Remove from in-memory list while preserving order
        st.session_state['chat_threads'] = [
            t for t in st.session_state['chat_threads'] if t != thread_id
        ]
        if st.session_state.get('thread_id') == thread_id:
            # If we deleted the active thread, start a fresh empty chat
            clear_chat()
        # Immediately refresh UI to reflect removal
        st.rerun()

# --------------------------
# Display Chat Messages
# --------------------------

# When a PDF was just uploaded, show "Analyzing PDF" in chat and run ingest there
if uploaded_pdf is not None:
    with st.chat_message("assistant"):
        with st.status("Analyzing PDF…", state="running"):
            try:
                info = ingest_pdf(
                    uploaded_pdf.read(),
                    st.session_state["thread_id"],
                    filename=uploaded_pdf.name,
                )
                st.success(
                    f"**{info.get('filename', 'PDF')}** loaded "
                    f"({info.get('documents', 0)} pages, {info.get('chunks', 0)} chunks). "
                    "Answers in this chat will use only this document."
                )
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# --------------------------
# User Input
# --------------------------

user_input = st.chat_input("Type your message here.")

if user_input and user_input.strip():
    # Only save/track a conversation once it has at least one message
    add_thread(st.session_state['thread_id'])

    # Add user message to session state
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Prepare config for chatbot
    config = {'configurable': {'thread_id': st.session_state['thread_id']},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn"
    }

    # When user has uploaded a PDF for this thread, answer only from it using RAG
    thread_id = st.session_state["thread_id"]
    if thread_has_document(thread_id):
        rag_only_system = (
            "The user has uploaded a PDF for this conversation. You MUST answer ONLY using the rag_tool. "
            "Do not use search, calculator, or stock tools. Call rag_tool with the user's question and with thread_id "
            f"exactly: {thread_id!r}. Then answer based solely on the returned context; if the context is empty or irrelevant, say so."
        )
        messages_for_turn = [
            SystemMessage(content=rag_only_system),
            HumanMessage(content=user_input),
        ]
    else:
        messages_for_turn = [HumanMessage(content=user_input)]

    # Stream assistant response and show tools being used
    stream = chatbot.stream(
        {"messages": messages_for_turn},
        config=config,
        stream_mode="messages",
    )

    ai_message_text = ""
    with st.chat_message('assistant'):
        tool_placeholder = st.empty()
        answer_placeholder = st.empty()

        for message_chunk, metadata in stream:
            # Try to infer and show any tools being used for this turn.
            tool_names: list[str] = []

            # Common pattern: tools listed on AIMessage.tool_calls
            if isinstance(message_chunk, AIMessage):
                tool_calls = getattr(message_chunk, "tool_calls", None) or []
                for tc in tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name:
                        tool_names.append(name)

            # Fallback: generic "tool-like" message that exposes name/tool attrs
            if hasattr(message_chunk, "tool") or hasattr(message_chunk, "name"):
                name = getattr(message_chunk, "name", None) or getattr(message_chunk, "tool", None)
                if name and name not in tool_names:
                    tool_names.append(name)

            if tool_names:
                unique_names = sorted(set(tool_names))
                tool_placeholder.markdown(
                    "**Using tool(s):** " + ", ".join(f"`{n}`" for n in unique_names)
                )

            # Stream back only assistant text to the main answer area
            if isinstance(message_chunk, AIMessage):
                content = message_chunk.content or ""
                # content from LangChain can sometimes be non-string; ensure string
                if not isinstance(content, str):
                    content = str(content)
                ai_message_text += content
                answer_placeholder.markdown(ai_message_text)

    # Save assistant response
    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message_text}
    )
