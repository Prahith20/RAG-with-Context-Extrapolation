from langchain_community.document_loaders import PDFPlumberLoader,PyMuPDFLoader
import re
from unidecode import unidecode
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from collections import defaultdict
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def clean_text(text):
    # 1. Normalize unicode characters
    text = unidecode(text)

    # 2. Remove repeated dots or whitespace clutter
    text = re.sub(r'([!?.])(?:\s*\1)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    # 3. Remove page numbers or repeated headers/footers (basic heuristic)
    text = re.sub(r'Page\s+\d+|\d+\s+Page', '', text, flags=re.IGNORECASE)

    # 4. Remove any non-printable/control characters
    text = re.sub(r'[^\x20-\x7E]', '', text)

    # 5. Remove long runs of punctuation or special chars
    text = re.sub(r'[~`_*^=\\]+', '', text)

    # 6. Strip any leading/trailing spaces
    text = text.strip()

    return text

# Function to get neighboring pages' chunks
def get_neighboring_chunks(chunks_by_page, page_num):
    neighboring_chunks = []
    
    # Get chunks for the current page
    neighboring_chunks.extend(chunks_by_page.get(page_num, []))
    
    # Get chunks for the previous page
    if page_num - 1 > 0:
        neighboring_chunks.extend(chunks_by_page.get(page_num - 1, []))
    
    # Get chunks for the next 2 pages
    neighboring_chunks.extend(chunks_by_page.get(page_num + 1, []))
    neighboring_chunks.extend(chunks_by_page.get(page_num + 2, []))
    
    return neighboring_chunks


def load_chunk_embed_pdf(file):
    loader = PyMuPDFLoader(file)
    docs = loader.load()
    docs = [
    Document(
        page_content=clean_text(doc.page_content),
        metadata=doc.metadata
    )
    for doc in docs
    ]

    combined_text = ""
    for i, doc in enumerate(docs):
        combined_text += f"\n\n[Page {i + 1}]\n" + doc.page_content

    #text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents([combined_text])

    page_marker_regex = re.compile(r"\[Page (\d+)\]")

    last_known_page = None

    for chunk in chunks:
        match = page_marker_regex.search(chunk.page_content)
        if match:
            page_num = int(match.group(1))
            chunk.metadata["page"] = page_num
            last_known_page = page_num
        elif last_known_page is not None:
            chunk.metadata["page"] = last_known_page

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store 
    vector = FAISS.from_documents(chunks, embedder)

    # Input
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    return retriever, chunks


def generate_llm_response(retriever, chunks, query):

    # Define prompt
    prompt = """
        You are given a question and a list of raw texts extracted from a PDF. There might or might not be a correlation between the texts.
        Your task is to answer the question based on the information from raw texts.
        
        Context: {context}
        
        Question: {question}


        Instructions:
        Consider yourself as a helpful RAG assistant that answers questions.
        Find and use the information from the raw texts to answer the question.
        Answer the question in a direct way without referring to the texts.
        List the individual page number(s) of the information that you used to answer the question. Cite it at the end of your response.
        Ensure that the answer is coherent, logically structured, and uses clear language.
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    
    # Define llm
    #llama_llm = Ollama(model="llama3.1:latest")
    openai_llm = ChatOpenAI(
        model_name="gpt-4o",  # Or "gpt-3.5-turbo"
        temperature=0,
        verbose=True
    ) 

    llm_chain = LLMChain(
                    llm=openai_llm, 
                    prompt=QA_CHAIN_PROMPT, 
                    callbacks=None, 
                    verbose=True)
    
    # Create a dictionary to store chunks by page number
    chunks_by_page = defaultdict(list)
    # Organize chunks by their page number
    for chunk in chunks:
        page_num = chunk.metadata.get("page")
        if page_num is not None:
            chunks_by_page[page_num].append(chunk)
    
    retrieved_docs = retriever.invoke(query)

    # Get the neighboring chunks for each retrieved document
    neighboring_sets = []
    for doc in retrieved_docs:
        page_num = doc.metadata["page"]
        neighboring_chunks = get_neighboring_chunks(chunks_by_page, page_num)
        
        # Add the set of neighboring chunks to the list
        neighboring_sets.append(neighboring_chunks)

    # Combine the neighboring chunks into single text blocks
    neighboring_texts = []
    for chunks_set in neighboring_sets:
        combined_text = "\n".join([chunk.page_content for chunk in chunks_set])
        neighboring_texts.append(combined_text)


    # Get response from llm
    response = llm_chain.run({"context": neighboring_texts[:3], "question": query})
    
    return response

