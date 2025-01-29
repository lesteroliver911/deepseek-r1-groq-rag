import streamlit as st
import os
from main import GroqRAG, RAGConfig
from tempfile import NamedTemporaryFile
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(
    page_title="DeepSeek R1 RAG Basic",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Header styling */
    .stTitle {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        color: #1E3A8A;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Upload section in sidebar */
    .uploadSection {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        border: 2px dashed #CBD5E1;
    }
    
    /* Question input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #E2E8F0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Thinking process section */
    .thinking-process {
        background-color: #F8FAFC;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Answer section */
    .answer-section {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Alerts and messages */
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Success message */
    .success-message {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #A7F3D0;
        margin: 1rem 0;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
        margin-bottom: 0.5rem !important;
    }
    
    .sidebar-subtitle {
        font-size: 1rem;
        color: #4B5563;
    }
    
    /* Main content area */
    .main-content {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

def extract_thinking_and_answer(response: str):
    """Extract thinking process and final answer from the response"""
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
        return thinking, answer
    else:
        return None, response

@st.cache_resource
def initialize_rag():
    """Initialize RAG system with caching"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please set GROQ_API_KEY environment variable")
        st.stop()
    return GroqRAG(api_key=api_key)

def main():
    # Initialize RAG system
    rag = initialize_rag()
    
    # Sidebar content
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>", unsafe_allow_html=True)
        st.markdown("<h1 class='sidebar-title'>🤖 DeepSeek R1 RAG</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-subtitle'>Upload PDF & Ask Questions</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # File upload section
        st.markdown("<div class='uploadSection'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['pdf'])
        if not uploaded_file:
            st.markdown("### 📄 Drop PDF here", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Question input (only show if file is uploaded)
        if uploaded_file:
            question = st.text_input("💭 Ask a question", key="question_input")
    
    # Main content area
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    if uploaded_file:
        try:
            # Create a temporary file
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process PDF if not already processed
            if 'processed_file' not in st.session_state or st.session_state.processed_file != uploaded_file.name:
                with st.spinner("📚 Processing your PDF..."):
                    texts = rag.load_pdf(tmp_path)
                    rag.create_vector_stores(texts)
                    rag.create_ensemble_retriever(texts)
                st.session_state.processed_file = uploaded_file.name
                st.markdown(f"<div class='success-message'>✅ Successfully processed: {uploaded_file.name}</div>", unsafe_allow_html=True)
            
            if question:
                # Initialize placeholders for streaming
                thinking_placeholder = st.empty()
                answer_placeholder = st.empty()
                
                with st.spinner("🤔 Analyzing your document..."):
                    full_response = ""
                    current_section = ""
                    in_thinking = False
                    
                    for chunk in rag.query_streaming(question):
                        full_response += chunk
                        
                        thinking, answer = extract_thinking_and_answer(full_response)
                        
                        if thinking:
                            with thinking_placeholder.expander("🧠 View AI's Thought Process", expanded=False):
                                st.markdown(f"<div class='thinking-process'>{thinking}</div>", unsafe_allow_html=True)
                        
                        if answer:
                            answer_placeholder.markdown(f"<div class='answer-section'><h3>📝 Answer:</h3>{answer}</div>", unsafe_allow_html=True)
            
            # Cleanup temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logging.error(f"Error: {str(e)}", exc_info=True)
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #6B7280; background-color: #F3F4F6; border-radius: 1rem;'>
                <h2 style='color: #1E3A8A; margin-bottom: 1rem;'>Welcome to DeepSeek R1 RAG</h2>
                <p style='font-size: 1.1rem; margin-bottom: 1rem;'>👈 Start by uploading a PDF document in the sidebar</p>
                <p style='font-size: 0.9rem;'>Supported format: PDF</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
