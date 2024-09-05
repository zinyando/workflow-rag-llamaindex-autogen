import os
from dotenv import load_dotenv
from chromadb import Collection, PersistentClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from autogen import ConversableAgent

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "llama-3.1-8b-instant",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}

rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="You are a RAG chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

db = PersistentClient(path="./chroma_db")
chroma_collection: Collection = db.get_or_create_collection("my-docs-collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


class SetupEvent(Event):
    query: str


class CreatePromptEvent(Event):
    query: str


class GenerateReplyEvent(Event):
    query: str


class RAGFlow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> SetupEvent | CreatePromptEvent:
        count = chroma_collection.count()

        if count < 1:
            return SetupEvent(query=ev.query)

        print("Loading existing index...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        await ctx.set("index", index)
        return CreatePromptEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        print("Creating new index...")
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        await ctx.set("index", index)
        return StartEvent(query=ev.query)

    @step
    async def create_prompt(
        self, ctx: Context, ev: CreatePromptEvent
    ) -> GenerateReplyEvent:
        index = await ctx.get("index")
        query_engine = index.as_query_engine()
        user_input = ev.query
        result = query_engine.query(user_input)

        prompt = f"""
        Your Task: Provide a concise and informative response to the user's query, drawing on the provided context.

        Context: {result}
        User Query: {user_input}

        Guidelines:
        1. Relevance: Focus directly on the user's question.
        2. Conciseness: Avoid unnecessary details.
        3. Accuracy: Ensure factual correctness.
        4. Clarity: Use clear language.
        5. Contextual Awareness: Use general knowledge if context is insufficient.
        6. Honesty: State if you lack information.

        Response Format:
        - Direct answer
        - Brief explanation (if necessary)
        - Citation (if relevant)
        - Conclusion
        """

        await ctx.set("prompt", prompt)
        return GenerateReplyEvent(query=ev.query)

    @step
    async def generate_reply(self, ctx: Context, ev: GenerateReplyEvent) -> StopEvent:
        prompt = await ctx.get("prompt")
        reply = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
        return StopEvent(result=reply["content"])


async def main():
    print("Welcome to RAGbot! Type 'exit', 'quit', or 'bye' to end the conversation.")
    while True:
        user_input = input(f"\nUser: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"Goodbye! Have a great day!!")
            break

        workflow = RAGFlow(timeout=10, verbose=True)
        reply = await workflow.run(query=user_input)
        print(f"\nRAGbot: {reply}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
