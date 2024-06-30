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
from prompts import system_prompt
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# Define global LLM and embeddings
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=512)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def load_indices(folder):
    vector_storage_path = f"./storage/{folder}_vector"

    # Load existing indices directly
    vector_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=vector_storage_path)
    )

    return vector_index

# List of folder names where indices are stored
folders = ["partners", "pole_digital_information"]

# Load existing indices
indices = {}
for folder in folders:
    vector_index = load_indices(folder)
    indices[folder] = {
        "vector_index": vector_index
    }

# Create query engines
query_engines = {}
for folder, index_dict in indices.items():
    vector_query_engine = index_dict["vector_index"].as_query_engine(llm=llm, similarity_top_k=4)
    query_engines[folder] = {
        "vector_query_engine": vector_query_engine
    }

query_engine_tools = []
for folder, engines in query_engines.items():
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=engines["vector_query_engine"],
            metadata=ToolMetadata(
                name=f"{folder}_vector_tool",
                description=f"Vector search tool for {folder} documents."
                            "The information are in French language."
                            "Use a detailed plain text question as input to the tool."
            ),
        )
    )

# Define and configure the OpenAI Agent
openai_agent = OpenAIAgent.from_tools(
    query_engine_tools,
    llm=llm,
    system_prompt=system_prompt,
    verbose=True,
)

def agent(question):
    result = openai_agent.chat(question)
    return result