import openai
import os
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from dotenv import load_dotenv
import pypdf

load_dotenv()

llm = AzureOpenAI(
    engine="LC-csv",
    model="gpt-4", temperature=0.0
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    engine="text-embedding-ada-002",
)


service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

input_file_path = 'attention_is_all_you-need.pdf'

documents = SimpleDirectoryReader(
    input_files=[input_file_path]
).load_data()
index = VectorStoreIndex.from_documents(documents)

query = "explain Scaled Dot-Product Attention in short and bullets"
query_engine = index.as_query_engine()
answer = query_engine.query(query)
answer.response