import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Optional
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Validate API key
if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Please check your .env file.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ Error configuring Google API: {str(e)}")
    st.stop()

def convert_to_latex(text: str) -> str:
    """Convert mathematical expressions to LaTeX format."""
    # Common mathematical patterns
    patterns = {
        r'(\d+)\^(\d+)': r'\1^{\2}',  # Superscript
        r'(\d+)_(\d+)': r'\1_{\2}',   # Subscript
        r'sigma': r'\\sigma',          # Sigma
        r'alpha': r'\\alpha',          # Alpha
        r'beta': r'\\beta',            # Beta
        r'gamma': r'\\gamma',          # Gamma
        r'delta': r'\\delta',          # Delta
        r'epsilon': r'\\epsilon',      # Epsilon
        r'lambda': r'\\lambda',        # Lambda
        r'mu': r'\\mu',                # Mu
        r'pi': r'\\pi',                # Pi
        r'omega': r'\\omega',          # Omega
        r'\\frac': r'\\frac',          # Fraction
        r'\\sum': r'\\sum',            # Summation
        r'\\prod': r'\\prod',          # Product
        r'\\int': r'\\int',            # Integral
        r'\\sqrt': r'\\sqrt',          # Square root
        r'\\infty': r'\\infty',        # Infinity
        r'\\partial': r'\\partial',    # Partial derivative
        r'\\nabla': r'\\nabla',        # Nabla
        r'\\cdot': r'\\cdot',          # Dot product
        r'\\times': r'\\times',        # Times
        r'\\div': r'\\div',            # Division
        r'\\pm': r'\\pm',              # Plus minus
        r'\\leq': r'\\leq',            # Less than or equal
        r'\\geq': r'\\geq',            # Greater than or equal
        r'\\neq': r'\\neq',            # Not equal
        r'\\approx': r'\\approx',      # Approximately
        r'\\propto': r'\\propto',      # Proportional to
        r'\\in': r'\\in',              # Element of
        r'\\subset': r'\\subset',      # Subset
        r'\\cup': r'\\cup',            # Union
        r'\\cap': r'\\cap',            # Intersection
    }
    
    # Replace patterns in text
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

@st.cache_data
def get_pdf_text(pdf_doc) -> str:
    """Extract text from PDF document."""
    try:
        text = ""
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
        if not text.strip():
            st.warning("⚠️ The PDF appears to be empty or unreadable.")
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""

@st.cache_data
def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks for processing."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        chunks = splitter.split_text(text)
        if not chunks:
            st.warning("⚠️ No text chunks were generated. The document might be too short or empty.")
        return chunks
    except Exception as e:
        st.error(f"❌ Error splitting text: {str(e)}")
        return []

@st.cache_resource
def get_vector_store(chunks: List[str]) -> Optional[FAISS]:
    """Create vector store from text chunks."""
    try:
        if not chunks:
            return None
        embeddings = HuggingFaceEmbeddings()
        return FAISS.from_texts(chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

@st.cache_data
def generate_summary(text: str) -> str:
    """Generate summary from text using Gemini model."""
    try:
        if not text.strip():
            return "No text available for summarization."
        
        prompt = f"""
        You are an AI assistant specializing in creating detailed summaries of academic documents for literature reviews. 
        Your task is to summarize the document following these guidelines:

        1. Identify the main theories or concepts discussed.
        2. Summarize the key findings from relevant studies.
        3. Highlight areas of agreement or consensus in the research.
        4. Summarize the methodologies used in the research.
        5. Provide an overview of the potential implications of the research.
        6. Suggest possible directions for future research based on the current literature.
        7. If there is any architecture used then explain the architecture of model stepwise.
        
        Additionally, provide detailed explanations of the mathematical aspects of the research paper:
        
        8. Describe and explain the key mathematical models, theorems, or equations used in the paper.
        For each equation, please format it in LaTeX style using the following format:
        $equation$
        
        Document text:
        {text}
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        return "Error generating summary. Please try again."

@st.cache_resource
def get_qa_chain():
    """Create QA chain for question answering."""
    try:
        prompt_template = """
        You are an AI research assistant. Use the provided context from research papers to answer the question as accurately as possible. If the answer is not available in the context, respond with, "The information is not available in the provided context."

        Context: {context}
        Question: {question}
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"❌ Error creating QA chain: {str(e)}")
        return None

def process_document(pdf_doc) -> None:
    """Process uploaded document."""
    with st.spinner("Your Research Paper is getting processed..."):
        try:
            st.session_state.raw_text = get_pdf_text(pdf_doc)
            if not st.session_state.raw_text:
                st.error("❌ Failed to extract text from the PDF.")
                return
                
            text_chunks = get_text_chunks(st.session_state.raw_text)
            if not text_chunks:
                st.error("❌ Failed to create text chunks.")
                return

            st.session_state.vector_store = get_vector_store(text_chunks)
            if not st.session_state.vector_store:
                st.error("❌ Failed to create vector store.")
                return

            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")

def generate_document_summary() -> None:
    """Generate document summary."""
    with st.spinner("Your Summary is getting generated..."):
        try:
            if not st.session_state.raw_text:
                st.error("❌ No document text available for summarization.")
                return

            st.session_state.summary = generate_summary(st.session_state.raw_text)
            st.success("✅ Summary generated!")
        except Exception as e:
            st.error(f"❌ Error generating summary: {str(e)}")

def answer_question(question: str) -> Optional[str]:
    """Answer questions about the document."""
    if not st.session_state.vector_store:
        st.error("❌ Please upload a document first.")
        return None

    with st.spinner("Thinking..."):
        try:
            docs = st.session_state.vector_store.similarity_search(question)
            chain = get_qa_chain()
            if not chain:
                st.error("❌ Failed to create QA chain.")
                return None

            response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
            return response["output_text"]
        except Exception as e:
            st.error(f"❌ Error answering question: {str(e)}")
            return None

def display_summary_with_latex(summary: str):
    """Display the summary with LaTeX support for equations."""
    # Split the summary into sections
    sections = summary.split('\n\n')
    
    for section in sections:
        if section.strip():
            # Check if this is a main section header (starts with a number and dot)
            if re.match(r'^\d+\.', section.strip()):
                # Extract the header text without the number
                header_text = re.sub(r'^\d+\.\s*', '', section.strip())
                st.markdown(f"## {header_text}")
                continue
            
            # Split into paragraphs
            paragraphs = section.split('\n')
            for para in paragraphs:
                if para.strip():
                    # Check if this is a subheading (starts with a dash or bullet point)
                    if para.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                        st.markdown(f"**{para.strip().lstrip('-•* ')}**")
                        continue
                    
                    # Check for LaTeX-style equations
                    if '$' in para:
                        # Extract and display equations
                        equations = re.findall(r'\$(.*?)\$', para)
                        for eq in equations:
                            st.latex(eq)
                        # Display the rest of the paragraph without equations
                        text = re.sub(r'\$.*?\$', '', para)
                        if text.strip():
                            st.write(text)
                    else:
                        st.write(para)

def main():
    st.title("📄 Understand Any Research Paper")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    with st.sidebar:
        st.header("Document Upload")
        pdf_doc = st.file_uploader("Upload your Research Documentation (PDF)", type="pdf")
        
        if pdf_doc:
            process_document(pdf_doc)
            if st.button("Generate Document Summary"):
                generate_document_summary()

    if st.session_state.summary:
        st.header("Document Summary (Literature Review Style)")
        display_summary_with_latex(st.session_state.summary)
    
    st.header("Ask research-related questions about the document")
    question = st.text_input("Enter your question:")
    
    if question:
        answer = answer_question(question)
        if answer:
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
