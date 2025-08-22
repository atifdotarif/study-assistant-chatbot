import os
import uuid
from dotenv import load_dotenv
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -----------------------------
# Environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Model
# -----------------------------
def make_openai_model(temperature: float = 0.7):
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)


# -----------------------------
# Structured output + prompt
# -----------------------------
response_schemas = [
    ResponseSchema(name="answer", description="Helpful answer to the student's query"),
    ResponseSchema(name="sources", description="List of sources or references if available"),
    ResponseSchema(name="difficulty", description="Difficulty level: easy/medium/hard"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
FORMAT_INSTRUCTIONS = output_parser.get_format_instructions()

PROMPT_OPENAI = ChatPromptTemplate.from_messages([
    ("system", "You are a study assistant chatbot. Help students understand concepts with clarity."),
    ("system", "Keep track of the conversation and use past context to improve answers."),
    ("system", "Always return output in structured format."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}\n\nFormat Instructions:\n{format_instructions}"),
])


# -----------------------------
# Memory store (per-session)
# -----------------------------
session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# -----------------------------
# Build chain factory
# -----------------------------
def build_chain(temperature: float = 0.7):
    llm = make_openai_model(temperature)
    chain = PROMPT_OPENAI | llm | output_parser

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return chain_with_memory


# -----------------------------
# Gradio handlers
# -----------------------------
def start_session():
    return str(uuid.uuid4())


def chat_step(user_msg, history, session_id, temperature):
    if not user_msg:
        return history, history

    chain = build_chain(float(temperature))

    try:
        inputs = {"question": user_msg, "format_instructions": FORMAT_INSTRUCTIONS}

        result = chain.invoke(
            inputs,
            config={"configurable": {"session_id": session_id}},
        )

        answer = result.get("answer", "(no answer)")
        sources = result.get("sources", "N/A")
        difficulty = result.get("difficulty", "N/A")

        decorated = f"{answer}\n\n📚 Sources: {sources}\n📊 Difficulty: {difficulty}"

        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": decorated}
        ]
        return history, history

    except Exception as e:
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"⚠️ Parser error: {e}"}
        ]
        return history, history


def reset_session():
    sid = start_session()
    return sid, []


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Study Assistant Chatbot") as demo:
    gr.Markdown("""
    # Study Assistant Chatbot
    - Uses **LangChain** with **memory** (per session)
    - Powered by **OpenAI GPT**
    - Returns **structured outputs** (answer, sources, difficulty)
    """)

    with gr.Row():
        temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
        session_id = gr.State(start_session())
        new_session_btn = gr.Button("New Session", variant="secondary")

    chatbox = gr.Chatbot(type="messages", height=450, label="Chat History")
    msg = gr.Textbox(placeholder="Ask a study question...", label="Your Message")
    send = gr.Button("Send", variant="primary")
    clear = gr.Button("Clear Chat")

    # Wire actions
    send.click(
        chat_step,
        inputs=[msg, chatbox, session_id, temperature],
        outputs=[chatbox, chatbox],
    )
    msg.submit(
        chat_step,
        inputs=[msg, chatbox, session_id, temperature],
        outputs=[chatbox, chatbox],
    )

    def _new_session_and_clear():
        sid, _ = reset_session()
        return sid, []

    new_session_btn.click(_new_session_and_clear, outputs=[session_id, chatbox])
    clear.click(_new_session_and_clear, outputs=[session_id, chatbox])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
