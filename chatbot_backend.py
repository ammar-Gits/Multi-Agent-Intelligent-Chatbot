from __future__ import annotations

from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import os
import requests
import random
import tempfile

load_dotenv()

# Per-thread PDF retrievers and metadata (thread_id -> retriever / metadata)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0.7
# )

# model = ChatHuggingFace(llm=llm)

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)

search_tool = DuckDuckGoSearchRun(region="us-en")


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        tid = str(thread_id)
        _THREAD_RETRIEVERS[tid] = retriever
        _THREAD_METADATA[tid] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[tid].copy()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool 
def calculator(first_num:float , second_num:float, operation: str)->dict:
    """
     Perform a basic arithmetic operation on two numbers.
     Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num + second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation."}
        
        return {"first_num":first_num, "second_num":second_num, "operation":operation, "result":result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_Stock_price(symbol: str)->dict:
    """
     Fetch latest stock price for a given symbol (e.g 'AAPL', 'TSLA')
     using AlphaVantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CKH7TKNKTXBRY6HM9"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the PDF document for this chat thread.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored document.
    Always include the thread_id when calling this tool.
    """
    tid = str(thread_id) if thread_id else None
    retriever = _THREAD_RETRIEVERS.get(tid) if tid else None
    if not retriever:
        return {
            "query": query,
            "error": "No PDF has been uploaded for this conversation. Ask the user to upload a PDF first.",
            "context": [],
            "metadata": [],
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    meta = [doc.metadata for doc in result]
    return {"query": query, "context": context, "metadata": meta}

def chat_node(state: ChatState):
    """LLM node that may answer or call a certain tool"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tools = [get_Stock_price, search_tool, calculator, rag_tool]
llm_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread = False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    """
    Return all thread_ids ordered by most recently updated (latest first).
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(rowid) DESC
        """
    )
    rows = cursor.fetchall()
    return [r[0] for r in rows]

def delete_thread(thread_id: str) -> None:
    """Permanently delete a conversation (all checkpoints and writes) by thread_id."""
    tid = str(thread_id)
    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)
    with conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
