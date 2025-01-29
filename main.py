from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import logging
import os
from tqdm import tqdm
from dataclasses import dataclass
import textwrap
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, field_validator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set tokenizers parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAGConfig(BaseModel):
    """Configuration for the RAG system"""
    chunk_size: int = Field(default=2000)
    chunk_overlap: int = Field(default=1000)
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "deepseek-r1-distill-llama-70b"  # Model for completions and reranking
    max_length: int = Field(default=2096, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    top_k_results: int = Field(default=3, gt=0)
    use_reranking: bool = Field(default=True)
    memory_window: int = Field(default=5, gt=0)  # Number of conversations to remember
    
    @field_validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        return v

class GroqLLM(LLM):
    client: Groq
    model_name: str
    temperature: float
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content

class GroqRAG:
    def __init__(self, api_key: str, config: RAGConfig = None):
        """Initialize the RAG system with Groq and local embeddings"""
        self.config = config or RAGConfig()
        self.client = Groq(api_key=api_key)
        self.llm = GroqLLM(
            client=self.client, 
            model_name=self.config.llm_model,
            temperature=self.config.temperature
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ".", ",", " "],
            keep_separator=True
        )
        self.vector_store = None
        self.faiss_path = "faiss_store"
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.memory = ConversationBufferMemory(k=self.config.memory_window)
        self.document_metadata = {}  # Store metadata about loaded documents

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and process a PDF document"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError("File must be a PDF")
            
            loader = PyPDFLoader(pdf_path)
            try:
                documents = loader.load()
                
                # Print initial content for debugging
                logging.info(f"Loaded {len(documents)} pages from PDF")
                
                # Filter out empty pages and headers/footers
                filtered_docs = []
                for i, doc in enumerate(documents):
                    content = doc.page_content.strip()
                    
                    # Split content into lines and filter
                    lines = content.split('\n')
                    filtered_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip if line is empty or contains common metadata patterns
                        if (line and 
                            len(line) > 10 and  # Minimum meaningful content length
                            not any(skip in line.lower() for skip in [
                                'downloaded from', 'wiley online library', 
                                'terms and conditions', 'creative commons',
                                'http://', 'https://', 'www.', '.com'
                            ])):
                            filtered_lines.append(line)
                    
                    # Only add pages with meaningful content
                    filtered_content = '\n'.join(filtered_lines)
                    if len(filtered_content.strip()) > 100:  # Minimum page content length
                        doc.page_content = filtered_content
                        filtered_docs.append(doc)
                        logging.info(f"Page {i+1} content length: {len(filtered_content)}")
                
                if not filtered_docs:
                    raise ValueError("No valid content found in PDF after filtering. The PDF might be empty or contain only metadata.")
                
                texts = self.text_splitter.split_documents(filtered_docs)
                logging.info(f"Created {len(texts)} text chunks from {len(filtered_docs)} pages")
                
                return texts
                
            except Exception as e:
                raise ValueError(f"Failed to parse PDF file: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error loading PDF: {str(e)}")
            raise

    def create_vector_stores(self, texts: List[Document], save_to_disk: bool = True):
        """Create FAISS vector store"""
        try:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            if save_to_disk:
                self.vector_store.save_local(self.faiss_path)
            logging.info("Created FAISS vector store")
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            raise

    def create_ensemble_retriever(self, texts: List[Document]):
        """Create hybrid retriever combining vector and keyword search with reranking"""
        try:
            # Split text content for BM25
            text_contents = [doc.page_content for doc in texts]
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_texts(text_contents)
            self.bm25_retriever.k = 2
            
            # Create FAISS retriever
            faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
            
            # Create base ensemble retriever
            base_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            # Add reranking if enabled
            if self.config.use_reranking:
                llm_chain_extractor = LLMChainExtractor.from_llm(self.llm)
                self.ensemble_retriever = ContextualCompressionRetriever(
                    base_compressor=llm_chain_extractor,
                    base_retriever=base_retriever
                )
            else:
                self.ensemble_retriever = base_retriever
            
            logging.info("Created ensemble retriever with reranking")
        except Exception as e:
            logging.error(f"Error creating ensemble retriever: {str(e)}")
            raise

    def _text_wrap(self, text: str, width: int = 120) -> str:
        """Wrap text for better formatting"""
        return textwrap.fill(text, width=width)

    def format_chat_prompt(self, query: str, context: List[str]) -> List[Dict[str, str]]:
        """Format the chat prompt with context and conversation history"""
        system_prompt = """You are a helpful AI assistant. Your task is to:
        1. Answer questions based on the provided context
        2. If you don't know the answer or can't find it in the context, say so
        3. Always cite your sources by referring to the specific parts of the document you used
        4. Provide clear, direct answers without showing your thinking process
        5. Do not use any XML-like tags in your response
        6. Keep responses concise and focused"""
        
        # Get conversation history
        history = self.memory.load_memory_variables({})
        
        formatted_context = "\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(context))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Previous conversation:\n{history.get('history', '')}\n\nContext:\n{formatted_context}\n\nQuestion: {query}"}
        ]
        return messages

    def query(self, question: str) -> str:
        """Query the RAG system using ensemble retrieval"""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store initialized. Please load documents first.")

            # Use ensemble retriever with configured number of results
            docs = self.ensemble_retriever.get_relevant_documents(
                question, 
                k=self.config.top_k_results
            )
            context = [doc.page_content for doc in docs]
            
            # Log retrieved context for debugging
            for i, ctx in enumerate(context):
                logging.debug(f"Retrieved context {i}: {ctx[:200]}...")

            messages = self.format_chat_prompt(question, context)

            completion = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=1024,
                top_p=0.95
            )

            response = completion.choices[0].message.content
            
            # Update conversation memory
            self.memory.save_context(
                {"input": question},
                {"output": response}
            )

            return response

        except Exception as e:
            logging.error(f"Error during query: {str(e)}")
            raise

    def query_streaming(self, question: str, k: int = 3):
        """Stream the response from Groq"""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store initialized")
            
            docs = self.vector_store.similarity_search(question, k=k)
            context = [doc.page_content for doc in docs]
            messages = self.format_chat_prompt(question, context)

            stream = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=1524,
                top_p=0.95,
                stream=True
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            logging.error(f"Error during streaming query: {str(e)}")
            raise

def main():
    # Example usage
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY environment variable")

    # Initialize RAG system
    rag = GroqRAG(api_key=api_key)
    
    # Load PDF with better error handling
    while True:
        try:
            # Clear screen (optional)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            pdf_path = input("\nEnter the path to your PDF file (or 'exit' to quit): ")
            if pdf_path.lower() == 'exit':
                return
                
            texts = rag.load_pdf(pdf_path)
            
            # Create vector stores first
            rag.create_vector_stores(texts)
            
            # Initialize ensemble retriever after vector store is created
            rag.create_ensemble_retriever(texts)
            
            # Clear screen after loading
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\nPDF loaded successfully! You can now ask questions about: {pdf_path}")
            
            # Interactive query loop
            while True:
                try:
                    print("\n" + "="*50)
                    question = input("\nEnter your question (or 'exit' to quit): ").strip()
                    if not question:
                        continue
                    if question.lower() == 'exit':
                        return
                    
                    print("\nProcessing your question...\n")
                    answer = rag.query(question)
                    print("\nAnswer:", rag._text_wrap(answer))
                    
                except Exception as e:
                    print(f"\nError during query: {str(e)}")
                    print("Please try another question.")
            
            break  # Break the loop if PDF loads successfully
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different PDF file.\n")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
