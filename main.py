import time
import streamlit as st
from model import generate_response
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from model import model  # reuse the same ChatGroq instance


st.title(":blue[RAG-Based] Legal Guidance Assistant :streamlit:")
st.header("Project Overview:", divider="blue")
st.subheader("""*This project is a Retrieval-Augmented Generation (RAG) based AI chatbot designed to answer real-world legal questions using Indian laws and court cases.*
    It retrieves relevant legal documents from a curated knowledge base and generates grounded, contextual responses to help users understand legal procedures and rights. The knowledge base used by this system has been manually curated from legal texts on web and publicly available sources. It's been designed to be extendable, allowing additional legal documents and cases to be incorporated over time. While the current implementation is not intended for enterprise-scale deployment due to resource constraints, it serves as a robust learning and prototyping system for understanding Retrieval-Augmented Generation""")

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# RESET CONVERSATION BUTTON
with st.sidebar:
    st.markdown("### Conversation Controls")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.rerun()



def build_standalone_question(summary: str, user_question: str) -> str:
    #Build a standalone legal question using the rolling conversation summary.
    if not summary.strip():
        return user_question

    return (
        f"In the context of the following legal discussion: "
        f"{summary} "
        f"Answer the question: {user_question}"
    )


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["previous_summary", "question", "answer"],
    template="""
You are maintaining a short legal topic summary of an ongoing conversation.

Previous summary (may be empty):
{previous_summary}

Latest user question:
{question}

Assistant's answer:
{answer}

Rules:
- Update the summary to reflect the evolving legal topic.
- Keep it under 2-3 sentences.
- Use neutral, factual legal language.
- Do NOT give legal advice.
- Do NOT introduce new facts.
- Focus only on what the discussion is about.

Updated summary:
"""
)

def update_conversation_summary(previous_summary, user_question, assistant_answer):
    #Use the LLM to update a bounded semantic summary of the conversation.
    chain = (
        RunnablePassthrough.assign(
            previous_summary=lambda _: previous_summary,
            question=lambda _: user_question,
            answer=lambda _: assistant_answer
        )
        | SUMMARY_PROMPT
        | model
        | StrOutputParser()
    )

    return chain.invoke({})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Describe your legal issue...")


# HANDLE NEW MESSAGE
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    standalone_question = build_standalone_question(
        st.session_state.conversation_summary,
        user_input
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(standalone_question)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.session_state.conversation_summary = update_conversation_summary(
        st.session_state.conversation_summary,
        user_input,
        response
    )


st.sidebar.write("Conversation summary:")
st.sidebar.write(st.session_state.conversation_summary)


st.markdown(
    "<small><i>This tool provides informational guidance based on legal documents "
    "and is not a substitute for professional legal advice.</i></small>",
    unsafe_allow_html=True
)
