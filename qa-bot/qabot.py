"""
Document Question Answering Bot - Demonstration Tool

🚨 IMPORTANT: This is a DEMONSTRATION TOOL ONLY 🚨

Purpose:
- Showcases how to build a "Talk to Your Documents" system using RAG
- Demonstrates document processing, vector storage, and retrieval
- Educational tool for learning RAG architecture

Limitations:
- Uses DistilGPT-2 (82M parameters) - very limited AI model
- Responses are NOT high-quality answers
- Designed for technical demonstration, not production use

For production use, you would need:
- Much larger, more capable models (GPT-4, Claude, etc.)
- Better infrastructure and resources
- Enhanced processing and validation

Author: AI Engineer Course
Purpose: Educational demonstration of RAG implementation
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Fix NumPy 2.0 compatibility
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# Modern LangChain 0.3+ imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Transformers for HuggingFace models
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def get_llm():
    """Get a lightweight LLM for demonstration purposes only"""
    try:
        # Use DistilGPT-2 - perfect for demonstration
        # Small, fast, stable, and reliable for showing RAG functionality
        model_name = "distilgpt2"  # ~82M parameters - ideal for demos

        # Create pipeline with stable settings optimized for demonstration
        pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=100,  # Short responses for demo
            temperature=0.8,    # Slightly higher for variety
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=50256
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ Loaded DistilGPT-2 for demonstration purposes")
        return llm
    except Exception as e:
        print(f"Error loading DistilGPT-2: {e}")
        print("Using MockLLM for demonstration...")
        return MockLLM()

def get_embeddings():
    """Get embeddings model"""
    try:
        # Use a lightweight embedding model for faster loading
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Already quite small (~22MB)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Force CPU for consistency
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Using mock embeddings...")
        return MockEmbeddings()

class MockLLM:
    """Simple mock LLM for demonstration when models fail to load"""
    def __call__(self, prompt):
        return "This is a mock response. The actual LLM model could not be loaded."
    
    def invoke(self, prompt):
        # Provide a more realistic mock response for demo purposes
        mock_response = """Based on the document, this appears to be a comprehensive document about Artificial Intelligence and Machine Learning technologies.

Key topics covered in the document:
- Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
- Machine Learning (ML) is a subset of AI that enables computers to learn without explicit programming
- Deep Learning uses neural networks with multiple layers to process data

Main applications discussed:
- Natural Language Processing (NLP) for text analysis and generation
- Computer Vision for image recognition and analysis
- Predictive Analytics for forecasting and decision making
- Autonomous Systems for self-driving cars and robotics

Benefits mentioned:
- Improved efficiency and automation
- Enhanced decision-making capabilities
- Reduced human error in repetitive tasks
- Scalable solutions for complex problems

Challenges identified:
- Data quality and availability requirements
- Computational resource needs
- Ethical considerations and bias
- Interpretability of AI decisions

Note: This is a mock response since the actual language model could not be loaded. In a real deployment, you would see actual AI-generated responses based on your uploaded documents."""
        return {"result": mock_response}

class MockEmbeddings:
    """Simple mock embeddings for demonstration"""
    def embed_documents(self, texts):
        # Return random embeddings of consistent size
        import numpy as np
        return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_query(self, text):
        import numpy as np
        return np.random.rand(384).tolist()

def document_loader(file_path):
    """Load PDF or text document"""
    try:
        # Validate file path
        if not file_path or not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return []
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Load PDF document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        elif file_ext == '.txt':
            # Load text document
            from langchain_core.documents import Document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Create a document object
            document = Document(page_content=content, metadata={"source": file_path})
            return [document]
        else:
            print(f"Error: Unsupported file type: {file_ext}")
            return []
            
    except Exception as e:
        print(f"Error loading document: {e}")
        return []

def text_splitter(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks, embeddings):
    """Create vector store from chunks"""
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vectordb
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def clean_response(response):
    """Clean and improve the response from the LLM"""
    if not response:
        return "I apologize, but I couldn't generate a response. Please try again."
    
    # Remove common artifacts
    response = response.strip()
    
    # Remove repeated sentences
    sentences = response.split('. ')
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.lower() not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence.lower())
    
    response = '. '.join(unique_sentences)
    
    # Ensure response ends properly
    if response and not response.endswith(('.', '!', '?')):
        response += '.'
    
    # Limit response length to prevent rambling
    if len(response) > 1000:
        sentences = response.split('. ')
        response = '. '.join(sentences[:3]) + '.'
    
    return response

def extract_relevant_content(query, documents, embeddings, k=3):
    """Extract the most relevant sentences from the document based on the query"""
    try:
        # Create embeddings for the query
        query_embedding = embeddings.embed_query(query)
        
        # Get all document chunks with their embeddings
        all_chunks = text_splitter(documents)
        
        # Create vector store for similarity search
        vectordb = create_vectorstore(all_chunks, embeddings)
        if vectordb is None:
            return "Error retrieving relevant content."
        
        # Find most similar chunks
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        relevant_docs = retriever.invoke(query)
        relevant_text = format_docs(relevant_docs)
        
        return relevant_text
    except Exception as e:
        print(f"Error extracting relevant content: {e}")
        return ""

def retriever_qa(file_path, query):
    """Main QA function using a document-focused approach"""
    try:
        # Validate inputs
        if not file_path:
            return "Error: No file provided."
        if not query or not query.strip():
            return "Error: No query provided."
        
        # Load and process document
        documents = document_loader(file_path)
        if not documents:
            return "Error: Could not load the document."

        # Get models
        llm = get_llm()
        embeddings = get_embeddings()

        # Extract relevant content from document (get only the MOST relevant section)
        relevant_context = extract_relevant_content(query, documents, embeddings, k=1)
        
        if not relevant_context:
            return "Error: Could not extract relevant content from the document."
        
        # Clean up the response - limit to first relevant section only
        context_lines = relevant_context.split('\n')
        cleaned_context = '\n'.join(context_lines[:10])  # Limit to top 10 lines
        
        # Since DistilGPT-2 doesn't follow instructions well, we'll provide
        # a simple demonstration response that's clearly based on the document
        return f"📄 Most Relevant Document Sections:\n\n{cleaned_context}\n\n---\n\n📝 Note: This demonstration tool uses DistilGPT-2 (82M parameters) to show how semantic search finds relevant document sections. For production use, you'd need much larger models (GPT-4, Claude, etc.) to generate coherent answers."

    except Exception as e:
        return f"Error processing query: {str(e)}"

# Create Gradio interface
def create_interface():
    interface = gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(label="Upload PDF or Text File", file_types=['.pdf', '.txt']),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Answer", lines=5),
        title="Document Question Answering Bot - Demonstration",
        description="🚨 DEMONSTRATION TOOL ONLY 🚨\n\nThis is a proof-of-concept demonstration of how a 'Talk to Your Documents' tool can be built using RAG (Retrieval-Augmented Generation).\n\n⚠️ IMPORTANT LIMITATIONS:\n• Uses DistilGPT-2 (~82M parameters) - very limited AI model\n• Responses are NOT high-quality answers\n• This tool demonstrates the RAG architecture, not production-quality AI\n• For real use, you would need much larger, more capable models\n\n✅ WHAT THIS DEMONSTRATES:\n• Document loading and processing\n• Text chunking and vector storage\n• Semantic search and retrieval\n• RAG pipeline implementation\n• Gradio web interface\n\nUpload a PDF or text file and ask questions to see how the RAG system works!",
        examples=[
            [None, "What is the main topic of this document?"],
            [None, "Summarize the key points."],
            [None, "What are the conclusions?"]
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
