import streamlit as st
import ollama
import shutil
import os
from pathlib import Path
from core.config_loader import ConfigLoader
from core.ollama_client import OllamaClient
from core.database import DatabaseManager
from core.retrieval_engine import RetrievalEngine

# --- Page Config ---
st.set_page_config(
    page_title="Obsidian Brain 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Better Readability ---
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper functions ---
def get_ollama_models():
    try:
        models_info = ollama.list()
        return [m['model'] for m in models_info['models']]
    except Exception as e:
        return ["llama3:latest", "nomic-embed-text:latest"]

def reset_vector_db(db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

@st.dialog("📖 Full Document Viewer")
def view_full_document(file_path):
    """Opens a modal dialog showing the full markdown content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown(f"**File:** `{file_path}`")
        st.divider()
        st.markdown(content)
    except Exception as e:
        st.error(f"Could not read file: {e}")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Brain Configuration")
    available_models = get_ollama_models()
    
    # Defaults
    default_llm = next((i for i, m in enumerate(available_models) if "llama3" in m or "mistral" in m), 0)
    default_embed = next((i for i, m in enumerate(available_models) if "nomic" in m or "embed" in m), 0)

    # 1. Models
    llm_model = st.selectbox(
        "LLM Model", 
        available_models, 
        index=default_llm,
        help="The 'Brain' that generates the answer. Llama3 is recommended for speed and accuracy."
    )
    embed_model = st.selectbox(
        "Embedding Model", 
        available_models, 
        index=default_embed,
        help="The model that converts text to numbers (vectors). MUST be the same one used during indexing."
    )
    
    st.divider()
    
    # 2. Vault Settings
    st.subheader("📂 Vault & Database")
    
    default_cfg = ConfigLoader()
    default_vault = default_cfg.config['system'].get('vault_path', './vault')
    
    vault_path_input = st.text_input(
        "Obsidian Vault Path", 
        value=default_vault,
        help="Absolute path to your local Obsidian folder containing .md files."
    )
    
    chunk_size = st.number_input(
        "Chunk Size", 
        value=1000, 
        step=100,
        help="How many characters per text block. Larger = more context per block, but less specific."
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap", 
        value=200, 
        step=50,
        help="How many characters shared between adjacent blocks. Prevents cutting sentences in half."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Update Index", type="primary", help="Scans for new/modified files and adds them to the DB. Does NOT delete existing data."):
            with st.spinner("Updating..."):
                try:
                    cfg = ConfigLoader()
                    cfg.config['system']['vault_path'] = vault_path_input
                    cfg.config['system']['chunk_size'] = chunk_size
                    cfg.config['system']['chunk_overlap'] = chunk_overlap
                    cfg.config['system']['embed_model'] = embed_model
                    client = OllamaClient(cfg)
                    DatabaseManager(cfg, client).index_vault()
                    st.success("Updated!")
                except Exception as e: st.error(f"{e}")

    with col2:
        if st.button("🔥 Rebuild DB", type="secondary", help="Nuclear option: Deletes the entire database and re-reads every file from scratch."):
            st.session_state['confirm_reset'] = True

    if st.session_state.get('confirm_reset'):
        st.warning("⚠️ Delete DB and re-embed all files?")
        if st.button("✅ Yes, Overwrite"):
            with st.spinner("Rebuilding..."):
                cfg = ConfigLoader()
                reset_vector_db(cfg.get('system', 'chroma_path'))
                cfg.config['system']['vault_path'] = vault_path_input
                cfg.config['system']['chunk_size'] = chunk_size
                cfg.config['system']['embed_model'] = embed_model
                DatabaseManager(cfg, OllamaClient(cfg)).index_vault()
                st.success("Rebuilt!")
                st.session_state['confirm_reset'] = False
        if st.button("❌ Cancel"): st.session_state['confirm_reset'] = False

    st.divider()
    
    # 3. Retrieval Strategy
    st.subheader("🔍 Retrieval Strategy")
    strategy = st.selectbox(
        "Method", 
        ["hybrid", "hyde", "vector"], 
        index=0,
        help="• Hybrid: Searches Keywords (BM25) AND Meaning (Vector).\n• HyDE: Hallucinates an answer first, then finds matching notes.\n• Vector: Standard semantic search."
    )
    
    top_k = st.slider(
        "Top K", 1, 20, 5,
        help="How many document chunks to retrieve and feed to the LLM."
    )
    
    c1, c2 = st.columns(2)
    with c1: 
        web_fallback = st.toggle(
            "Web Search", 
            value=True,
            help="If enabled, searches DuckDuckGo when no local notes are found."
        )
    with c2: 
        use_crag = st.toggle(
            "CRAG", 
            value=False,
            help="(Experimental) Corrective RAG: Evaluates retrieved documents and discards irrelevant ones."
        )

# --- Main App ---
st.title("🧠 Obsidian Brain")

# Init Backend
cfg = ConfigLoader()
cfg.config['system']['vault_path'] = vault_path_input
cfg.config['system']['llm_model'] = llm_model
cfg.config['system']['embed_model'] = embed_model
cfg.config['retrieval']['strategy'] = strategy
cfg.config['retrieval']['top_k'] = top_k
cfg.config['retrieval']['web_fallback'] = web_fallback

# Show History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- PERSISTENT SOURCE VIEWER ---
# We check session state specifically to keep the expander open/available
# regardless of whether the user just hit enter or clicked a button.
if st.session_state.last_retrieved_docs:
    with st.expander(f"📚 Context from last query ({len(st.session_state.last_retrieved_docs)} chunks)", expanded=False):
        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            source = doc.metadata.get('filename', 'Unknown')
            file_path = doc.metadata.get('source', None)
            
            with st.container(border=True):
                col_text, col_btn = st.columns([0.85, 0.15])
                with col_text:
                    st.markdown(f"**{i+1}. {source}**")
                    st.caption(doc.page_content[:200].replace("\n", " ") + "...")
                
                with col_btn:
                    # Unique key is essential for buttons in loops
                    if file_path and st.button("📖", key=f"btn_{i}", help="Read full document"):
                        view_full_document(file_path)

# --- Chat Input ---
if prompt := st.chat_input("Ask your second brain..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        docs = []
        
        # 1. Retrieval
        with st.status("Thinking...", expanded=True) as status:
            try:
                client = OllamaClient(cfg)
                db = DatabaseManager(cfg, client)
                engine = RetrievalEngine(cfg, db, client)
                
                st.write(f"🔍 Searching via **{strategy}**...")
                docs = engine.execute_retrieval(prompt)
                
                # SAVE DOCS TO SESSION STATE (The Fix!)
                st.session_state.last_retrieved_docs = docs
                
                if docs: status.update(label="Context Retrieved!", state="complete", expanded=False)
                else: status.update(label="No local documents found.", state="error", expanded=False)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        
        # 2. Rerun to show the Source Expander immediately
        # (Optional: feels snappier if sources appear before text generation finishes)
        
        # 3. Generation
        context_text = "\n\n".join([d.page_content for d in docs])
        final_prompt = f"Answer strictly using this Context:\n{context_text}\n\nQuestion: {prompt}"
        
        try:
            stream = client.get_llm().stream(final_prompt)
            for chunk in stream:
                if chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Generation failed: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Force a rerun so the "Persistent Source Viewer" at the top updates with the new docs
    st.rerun()