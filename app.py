import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio
from prompts import system_prompt

nest_asyncio.apply()
load_dotenv()

# Define global LLM and embeddings
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=512)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def load_indices(folder):
    vector_storage_path = f"./storage/{folder}_vector"
    vector_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=vector_storage_path)
    )
    return vector_index

folders = ["partners", "pole_digital_information"]

indices = {}
for folder in folders:
    vector_index = load_indices(folder)
    indices[folder] = {"vector_index": vector_index}

query_engines = {}
for folder, index_dict in indices.items():
    vector_query_engine = index_dict["vector_index"].as_query_engine(llm=llm, similarity_top_k=4)
    query_engines[folder] = {"vector_query_engine": vector_query_engine}

query_engine_tools = []
for folder, engines in query_engines.items():
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=engines["vector_query_engine"],
            metadata=ToolMetadata(
                name=f"{folder}_vector_tool",
                description=f"Vector search tool for {folder} documents. The information is in French language. Use a detailed plain text question as input to the tool."
            ),
        )
    )

openai_agent = OpenAIAgent.from_tools(
    query_engine_tools,
    llm=llm,
    system_prompt=system_prompt,
    verbose=True,
)

def agent(question):
    resp = openai_agent.chat(question)
    return resp.response  # Extract the response text

# Streamlit interface
st.title("Assistant IA")

st.write("""
         Bonjour, je suis votre assistant du Pôle Digital.
         Comment puis-je vous aider?
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Entrez votre question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.spinner("Querying..."):
        response_text = agent(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})


