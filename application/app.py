from utils import load_chunk_embed_pdf, generate_llm_response
import tempfile
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename.lower()
    if file and filename.endswith(('.pdf', '.txt')):
        # Save the uploaded PDF to a temporary file
        extension = os.path.splitext(filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            global retriever, chunks
            retriever, chunks = load_chunk_embed_pdf(temp_file_path)

        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

        return jsonify({"status": "done"})

    return jsonify({"error": "Invalid file type, only text and pdf files allowed"}), 400

@app.route('/chat', methods=['POST'])
def process_text():
    content = request.json
    query = content.get('query', '')
    global retriever, chunks

    response = generate_llm_response(retriever, chunks, query)

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
