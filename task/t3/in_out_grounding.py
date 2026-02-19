import json
import os
import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


ABOUT_USER = """User:
    id: {id}
    about user: {about}
"""

DB_DIRECTORY = './chroma_langchain_db'


def retrieve_context(vector_store: Chroma, query: str, k: int = 20, score_threshold: float = 0.1) -> str:
    results = vector_store.similarity_search_with_relevance_scores(query, k=k, score_threshold=score_threshold)

    context_parts = []

    for doc, score in results:
        context_parts.append(ABOUT_USER.format(id=doc.id, about=doc.page_content))

    return '\n\n'.join(context_parts)


def augment_prompt(query: str, context: str) -> str:
    return USER_PROMPT.format(query=query, context=context)


def generate_answer(llm_client: AzureChatOpenAI, augmented_prompt: str) -> dict[str, list[str]]:
    parser = PydanticOutputParser(pydantic_object=GroupingResults)
    messages = [
        SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
        HumanMessage(augmented_prompt)
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
    ai_response = (prompt | llm_client | parser).invoke({})

    return {res.hobby: res.user_ids for res in ai_response.grouping_results}


async def validate_answer(user_client: UserClient, anwser: dict[str, list[str]]) -> str:
    results: dict[str, list[dict[str, Any]]] = {}

    for hobby, user_ids in anwser.items():
        results[hobby] = []

        for id_ in user_ids:
            try:
                user_data = await user_client.get_user(int(id_))
            except Exception as e:
                if '404' in str(e):
                    continue
                raise

            results[hobby].append(user_data)

        # remove if no users found
        if not results[hobby]:
            del results[hobby]

    return json.dumps(results)


async def init_db(vector_store: Chroma, user_client: UserClient) -> None:
    all_users = user_client.get_all_users()
    documents = [Document(u['about_me']) for u in all_users]
    ids = [str(u['id']) for u in all_users]
    print(f'Initial db load: add {len(documents)} users')
    await vector_store.aadd_documents(documents=documents, ids=ids)


async def update_db(vector_store: Chroma, user_client: UserClient) -> None:
    all_users = user_client.get_all_users()
    db_data = vector_store.get(include=[])

    all_users_ids = [str(u['id']) for u in all_users]

    users_to_add = [u for u in all_users if str(u['id']) not in db_data['ids']]
    ids_to_remove = [id_ for id_ in db_data['ids'] if id_ not in all_users_ids]

    if users_to_add:
        print(f'Add {len(users_to_add)} users')
        documents = [Document(u['about_me']) for u in users_to_add]
        ids = [str(u['id']) for u in users_to_add]
        await vector_store.aadd_documents(documents=documents, ids=ids)

    if ids_to_remove:
        print(f'Remove {len(users_to_add)} users')
        await vector_store.adelete(ids_to_remove)


async def main() -> None:
    embeddings=AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
    )
    llm_client=AzureChatOpenAI(
        model='gpt-4o',
        temperature=0.0,
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version='',
    )
    user_client = UserClient()


    db_exists = os.path.exists(DB_DIRECTORY)

    vector_store = Chroma(
        collection_name='example_collection',
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY,
        collection_metadata={'hnsw:space': 'cosine'}
    )

    if not db_exists:
        await init_db(vector_store, user_client)

    print('type question or \'exit\'')
    print('e.g. "I need people who love to go to photography"')

    while True:
        user_question = input('\n>')
        user_question = user_question.strip()

        if user_question == 'exit':
            break

        await update_db(vector_store, user_client)

        if context := retrieve_context(vector_store, query=user_question):
            augmented_prompt = augment_prompt(user_question, context)
            answer = generate_answer(llm_client, augmented_prompt)
            validated_answer = await validate_answer(user_client, answer)
            print(f'validated answer: {validated_answer}')
        else:
            print('no relevant data to provide an answer')



asyncio.run(main())
