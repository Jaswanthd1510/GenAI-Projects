### Personal RAG Assistant

Building a local RAG (Retrieval-Augmented Generation) system using Ollama, LangChain, and Streamlit while leveraging powerful LLMs.

```bash
personal_rag_assistant/
├── personal_rag_assistant.py  # Streamlit UI and session management
├── rag.py       # Core RAG logic (Embeddings, Vector Store, LLM)
├── config.py           # Configuration and constants
├── requirements.txt    # Python dependencies
└── data/               # (Auto-created) Local ChromaDB storage
```

### Installation

```bash
# Git Bash (Windows)
python -m venv rag-env
source rag-env/Scripts/activate
pip install -r requirements.txt

# (macOS / Linux)
# python3 -m venv rag-env
# source rag-env/bin/activate
```

### Ollama Installation and Model Downloads

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull gemma3
```

### To run the app locally

```python
streamlit run personal_rag_assistant.py
```