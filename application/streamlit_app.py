import streamlit as st
import requests

# ---- Backend Endpoints ----
UPLOAD_URL = "http://localhost:8000/upload"  # Replace with actual
CHAT_URL = "http://localhost:8000/chat"      # Replace with actual

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# ---- Session State Initialization ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_done" not in st.session_state:
    st.session_state.upload_done = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ---- Sidebar Controls ----
st.sidebar.title("⚙️ Options")
if st.sidebar.button("🔁 Reset for a New Document"):
    st.session_state.chat_history = []
    st.session_state.upload_done = False
    st.session_state.uploaded_file_name = None
    st.rerun()

# ---- Upload Document Phase ----
if not st.session_state.upload_done:
    st.title("📄 Upload Document for RAG")
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT)", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Uploading and processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(UPLOAD_URL, files=files)
                result = response.json()

                if result.get("status") == "done":
                    st.session_state.upload_done = True
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success("✅ Document processed. You can now chat!")
                    #st.rerun()
                else:
                    st.error("❌ Backend failed to process the document.")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

# ---- Chat Interface Phase ----
if st.session_state.upload_done:
    st.title("💬 Chat with Your Document")
    st.caption(f"Using document: `{st.session_state.uploaded_file_name}`")

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    # Chat input
    prompt = st.chat_input("Ask a question...")

    if prompt:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                response = requests.post(CHAT_URL, json={"query": prompt})
                response.raise_for_status()
                answer = response.json().get("answer", "🤖 No response.")
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
