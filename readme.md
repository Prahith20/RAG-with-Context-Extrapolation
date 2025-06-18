# RAG with Context Extrapolation

This project shows a slightly better retrieval method on top of a classical approach for a RAG application.

---

## The problem  
If a topic spans 2–3 pages in a document, and the vector db only returns the first or second page that the query matches the most, the LLM responsible for response generation loses context that might be useful.

---

## Solution  
After retrieving the query relevant chunks from the retriever, the chunks from the previous few pages and the next few pages are retrieved, and sent to the generator.  
It's a simple approach, but in certain cases, I found that it works better than using a reranker or contextual chunking.

---

## Step-by-Step approach

- Get the document  
- Extract text from all the pages  
- Concat the texts (can read and chunk in batches for very large PDFs as concatenating takes up a large space in memory)  
- Create overlapping chunks  
- Create embeddings and ingest to a vector database  
- With the user's query, get the relevant chunks  
- For each retrieved chunk, get the previous page and the next 2 pages (feel free to experiment with the number of preceding and subsequent pages)  
- Get a response from the generator  

---

## Note  
I have added a page number tag to each chunk, from the page it belongs, to prompt the generator to cite the page numbers of its response.  
Hence, the concatenation and chunking instead of chunking per page.

---

## Useful for

- Office policy documents  
- Learning materials for study  
- Cooking recipes, etc.

---

## Explore

- Explore the use cases in the `RAG with Context Extrapolation.ipynb` notebook.  
**or**  
- Run the application in a Streamlit app.

---

## Prerequisites

- An OpenAI API key and access to `gpt-4o` or `gpt-3.5-turbo` models. Save it in .env file in the application.
**or**  
- Use Ollama models to keep the inference private (Ex: `llama3.1:latest`)

---

## To run the application

```bash
# Clone the repository
git clone https://github.com/Prahith20/RAG-with-Context-Extrapolation

# Go to project directory
cd application

# Create a virtual environment and activate it
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# In a terminal, run the backend
python app.py

# In another terminal, run the frontend
streamlit run streamlit_app.py
```

---

The app will be available at: http://localhost:8501

---

## Note  
Depending on the use case, we can combine this with re-ranking methods, multi-query techniques, or a different vector database.
