import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List
from langchain.docstore.document import Document

import requests

from .cache import Cache
from .config import cfg
# for embeddings
import os
import contextlib
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

CACHE_LENGTH = int(cfg.get("CACHE_LENGTH"))
CACHE_PATH = Path(cfg.get("CACHE_PATH"))
REQUEST_TIMEOUT = int(cfg.get("REQUEST_TIMEOUT"))
DISABLE_STREAMING = str(cfg.get("DISABLE_STREAMING"))

HTTP_REFERER = str(cfg.get("HTTP_REFERER"))
APP_TITLE = str(cfg.get("APP_TITLE"))

EMBEDDING_TEMPLATE = """Answer the question based on the following context:

{context}

Question: {question}
"""
EXCLUDED_NAMES = {"node_modules"}
EXCLUDED_START_PATTERNS = {".", "_"}
EXCLUDED_EXTENSIONS = {".obj", ".bin", ".glb", ".stl", ".gltf", ".ttf", ".woff"} 

class OpenAIClient:
    cache = Cache(CACHE_LENGTH, CACHE_PATH)

    def __init__(self, api_host: str, api_key: str) -> None:
        self.__api_key = api_key
        self.api_host = api_host

    def format_docs(self, docs):
        """
        Format the documents for context retrieval.
        :param docs: List of documents.
        :return: String containing all documents.
        """
        return "\n\n".join([d.page_content for d in docs])

    def load_vectorstore(self, root_dir):
        """
        Load a whole directory into a vector store index. If a '.vectorstore' directory 
        exists within the given directory, load the index from there. Otherwise, 
        create a new index, store it in '.vectorstore', and return it.

        :param directory_path: Path to the directory containing documents to index.
        :return: The loaded or created vector store index.
        """
        persist_path = os.path.join("/tmp", "shell_gpt", "vectorstore")
        if self.file_context:
            persist_path = os.path.join(root_dir, ".vectorstore")

        if os.path.exists(persist_path):
            vectorstore = Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings())
        else:
            vectorstore = self.create_vectorstore(persist_path)
        self.vectorstore = vectorstore
        return vectorstore

    def create_vectorstore(self, persist_path):
        tree_str = ""
        prefix = " "  # Prefix used for each level of depth
        root_dir = persist_path.rsplit("/", 1)[0]
        vs_dir = persist_path.rsplit("/", 1)[1]
        os.makedirs(root_dir, exist_ok=True)
        persist_path = os.path.join(root_dir, vs_dir)
        file_docs = []
        
        vectorstore = Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings())

        if self.file_context:
            def add_to_tree(path, depth):
                """Add path to the tree with the given depth."""
                nonlocal tree_str
                indent = prefix * depth
                tree_str += f"{indent}{os.path.basename(path)}\n"
            
            for path, dirs, files in os.walk(root_dir):
                depth = path.count(os.sep) - root_dir.count(os.sep)
                # Filter directories
                dirs[:] = [d for d in dirs if d not in EXCLUDED_NAMES and not any(d.startswith(pat) for pat in EXCLUDED_START_PATTERNS)]
                # Filter files
                files = [f for f in files if os.path.splitext(f)[1].lower() not in EXCLUDED_EXTENSIONS]
                files = [f for f in files if f not in EXCLUDED_NAMES and not any(f.startswith(pat) for pat in EXCLUDED_START_PATTERNS)]
                # Add dirs and files to the tree, maintaining the depth
                for name in dirs + files:
                    add_to_tree(os.path.join(path, name), depth)
                    file_path = os.path.join(path, name)
                    try:
                        docs = UnstructuredFileLoader(file_path, unstructured_kwargs="").load()
                        if len(docs):
                            #print(f"Loaded {len(docs)} documents from {file_path}")
                            file_docs.append(docs)
                    except:# Exception as e: 
                        pass
                        #print(f"An error occurred: {e}")
        
            file_structure = "Present working directory structure:\n" + tree_str
            tree_doc = Document(page_content=file_structure, metadata={"source": "local"})
            vectorstore.add_documents([tree_doc])

            for docs in file_docs:
                try:
                    vectorstore.add_documents(docs)
                except:
                    #print("Error adding file documents to vectorstore: {}".format(docs))
                    continue
        return vectorstore

    def make_history(self, docs: List[Document]) -> str:
        history = ''
        for doc in docs:
            #print(doc)
            try:
                if doc.metadata["role"]:
                    timestamp = doc.metadata["timestamp"]
                    history += f"{timestamp} {doc.metadata['role']}: {doc.page_content}\n"
            except:
                docs.remove(doc)
                continue
        return history

    def handle_context(self, local_path, messages):
        """
        Add context to the last message in the list of messages.
        :param local_path: Path to the directory containing documents to index.
        :param messages: List of dict with messages and roles.
        :return: None.
        """
        context = ""
        question = messages[-1]["content"]
        self.load_vectorstore(local_path)

        embedding_vector = OpenAIEmbeddings().embed_query(question)
        vs_len = len(self.vectorstore.get()["ids"])
#         n_results = min(vs_len, 2)
        docs = []
#         if n_results > 0:
#             docs = self.vectorstore.similarity_search_by_vector(embedding_vector, n_results)

#         n_results = min(vs_len, 10)
        #print("n:", n_results)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    last_n_docs = self.vectorstore.as_retriever(
#                     search_kwargs={
#                     "filter": {
#                         "role": {
#                             "$in": ["user", "assistant"]
#                             }
#                         }
#                     }
                    ).get_relevant_documents('')#, kwargs={"n_results": n_results})
        #print("last:", len(last_n_docs))
        # sort by timestamp
        last_n_docs.sort(key=lambda x: x.metadata["timestamp"])

        if len(last_n_docs) > 0:
            history = self.make_history(last_n_docs)
            context += history + "\n\n" + question
        
        if self.file_context:
            # file tree docs
            tree_docs = self.vectorstore.as_retriever(search_kwargs={"filter": {"source": "local"}}).get_relevant_documents("")
            docs = tree_docs + docs
            context += "\n\n" + self.format_docs(docs)
            messages[-1]["content"] = EMBEDDING_TEMPLATE.format(context=context, question=question)
        else:
            context += "\n\n" + self.format_docs(docs)
            messages[-1]["content"] = context + "\n\n" + question

        # store question in vectorstore
        question_doc = Document(page_content=question, metadata={"role": "user", "timestamp": str(datetime.now())})
        self.vectorstore.add_documents([question_doc])

    @cache
    def _request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        top_probability: float = 1,
        file_context: bool = False,
    ) -> Generator[str, None, None]:
        """
        Make request to OpenAI API, read more:
        https://platform.openai.com/docs/api-reference/chat

        :param messages: List of messages {"role": user or assistant, "content": message_string}
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301
        :param temperature: Float in 0.0 - 2.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :return: Response body JSON.
        """
        # embeddings
        self.file_context = file_context
        self.handle_context(os.getcwd(), messages)

        stream = DISABLE_STREAMING == "false"
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_probability,
            "stream": stream,
        }
        endpoint = f"{self.api_host}/v1/chat/completions"
        response = requests.post(
            endpoint,
            # Hide API key from Rich traceback.
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": APP_TITLE,
            },
            json=data,
            timeout=REQUEST_TIMEOUT,
            stream=stream,
        )
        response.raise_for_status()
        # TODO: Optimise.
        # https://github.com/openai/openai-python/blob/237448dc072a2c062698da3f9f512fae38300c1c/openai/api_requestor.py#L98
        str_response = ''
        if not stream:
            data = response.json()
            yield data["choices"][0]["message"]["content"]  # type: ignore
            str_response = data["choices"][0]["message"]["content"]  # type: ignore
            return
        for line in response.iter_lines():
            # openrouter first line
            if line.startswith(b": OPENROUTER PROCESSING"):
                continue
            data = line.lstrip(b"data: ").decode("utf-8")
            if data == "[DONE]":  # type: ignore
                break
            if not data:
                continue
            data = json.loads(data)  # type: ignore
            delta = data["choices"][0]["delta"]  # type: ignore
            if "content" not in delta:
                continue
            yield delta["content"]
            str_response += delta["content"]

        # store response in vectorstore
        response_doc = Document(page_content=str_response, metadata={"role": "assistant", "timestamp": str(datetime.now())})
        self.vectorstore.add_documents([response_doc])

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        top_probability: float = 1,
        caching: bool = True,
        file_context: bool = False,
    ) -> Generator[str, None, None]:
        """
        Generates single completion for prompt (message).

        :param messages: List of dict with messages and roles.
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301.
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :param caching: Boolean value to enable/disable caching.
        :return: String generated completion.
        """
        yield from self._request(
            messages,
            model,
            temperature,
            top_probability,
            caching=caching,
            file_context=file_context,
        )
