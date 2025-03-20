import streamlit as st
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import arxiv
import logging
from typing import List, Tuple, Optional
import time
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the model and tokenizer to avoid reloading on every run
@st.cache_resource
def load_scibert_model() -> Tuple[Optional[AutoModel], Optional[AutoTokenizer]]:
    try:
        model_name = "allenai/scibert_scivocab_uncased"
        logger.info(f"Loading SciBERT model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading SciBERT model: {str(e)}")
        return None, None

# Cache the summarization pipeline
@st.cache_resource
def load_summarizer() -> Optional[pipeline]:
    try:
        logger.info("Loading summarization model")
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"❌ Error loading summarization model: {str(e)}")
        return None

# Load the model, tokenizer, and summarizer
scibert_model, tokenizer = load_scibert_model()
summarizer = load_summarizer()

# Function to get SciBERT embeddings for a given text
def get_scibert_embedding(text: str) -> Optional[np.ndarray]:
    try:
        if not text.strip():
            return None
            
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = scibert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {str(e)}")
        return None

# Fetch ArXiv papers using their API
def fetch_arxiv_papers(query: str, max_results: int = 5, categories: Optional[str] = None) -> List[arxiv.Result]:
    try:
        category_query = f" AND cat:{categories}" if categories else ""
        full_query = query + category_query
        logger.info(f"Searching arXiv with query: {full_query}")
        
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
        
        if not results:
            st.warning("⚠️ No papers found for the given query.")
            
        return results
    except Exception as e:
        st.error(f"❌ Error fetching papers from arXiv: {str(e)}")
        return []

# Function to extract excerpts related to the concept from papers
def extract_paper_excerpts(papers: List[arxiv.Result]) -> List[Tuple[str, str, str, str, datetime]]:
    try:
        excerpts = []
        for paper in papers:
            abstract = paper.summary
            if not abstract:
                continue
                
            # Use summarization model to create concise excerpts
            summary = summarizer(abstract, max_length=50, min_length=30, do_sample=False)
            excerpts.append((
                paper.title,
                summary[0]['summary_text'],
                paper.pdf_url,
                paper.entry_id,
                paper.published
            ))
        return excerpts
    except Exception as e:
        st.error(f"❌ Error extracting paper excerpts: {str(e)}")
        return []

# List closely related papers by calculating similarity between SciBERT embeddings
def list_related_papers_with_excerpts(
    concept: str,
    max_results: int = 5,
    categories: str = "cs.LG"
) -> List[Tuple[str, str, str, str, datetime]]:
    try:
        if not concept.strip():
            st.error("❌ Please enter a valid search concept.")
            return []
            
        papers = fetch_arxiv_papers(concept, max_results=max_results, categories=categories)
        
        if not papers:
            return []
        
        # Get SciBERT embedding for the concept
        concept_embedding = get_scibert_embedding(concept)
        if concept_embedding is None:
            st.error("❌ Failed to generate embeddings for the concept.")
            return []
        
        related_papers = []
        for paper in papers:
            paper_text = paper.title + " " + paper.summary
            paper_embedding = get_scibert_embedding(paper_text)
            if paper_embedding is None:
                continue
                
            # Calculate similarity score
            similarity_score = cosine_similarity([concept_embedding], [paper_embedding]).flatten()[0]
            related_papers.append((paper, similarity_score))
        
        # Sort papers by similarity score and publication date
        related_papers.sort(key=lambda x: (x[1], x[0].published), reverse=True)
        
        # Extract excerpts from papers
        excerpts = extract_paper_excerpts([paper[0] for paper in related_papers])
        
        return excerpts
    except Exception as e:
        st.error(f"❌ Error finding related papers: {str(e)}")
        return []

def display_paper_card(title: str, excerpt: str, url: str, paper_id: str, published_date: datetime):
    st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{title}</div>
            <div class="paper-excerpt">{excerpt}</div>
            <div style="margin-top: 1rem;">
                <a href="{url}" target="_blank" class="paper-link">📄 Read Paper</a>
                <span style="float: right; color: #666; font-size: 0.9rem;">
                    Published: {published_date.strftime('%Y-%m-%d')}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Streamlit app for listing related papers and excerpts
def main():
    st.title("🔍 Find Related Research Papers")
    
    # Check if models are loaded
    if scibert_model is None or tokenizer is None:
        st.error("❌ Failed to load required models. Please try refreshing the page.")
        return
        
    if summarizer is None:
        st.error("❌ Failed to load summarization model. Please try refreshing the page.")
        return
    
    # Search input
    concept = st.text_input("Enter a research concept or topic", placeholder="e.g., transformer architecture in deep learning")
    
    # Category selection
    categories = {
        "Computer Science": "cs.LG",
        "Machine Learning": "cs.LG",
        "Artificial Intelligence": "cs.AI",
        "Neural Networks": "cs.NE",
        "All Categories": None
    }
    selected_category = st.selectbox("Select Category", list(categories.keys()))
    category = categories[selected_category]
    
    # Number of results
    max_results = st.slider("Number of Results", min_value=1, max_value=10, value=5)
    
    if st.button("🔍 Search Papers", key="search_button"):
        if not concept.strip():
            st.error("❌ Please enter a research concept or topic.")
            return
            
        with st.spinner("Searching for related papers..."):
            try:
                related_papers = list_related_papers_with_excerpts(
                    concept,
                    max_results=max_results,
                    categories=category
                )
                
                if not related_papers:
                    st.warning("⚠️ No related papers found matching your criteria.")
                else:
                    st.markdown(f"### Found {len(related_papers)} Related Papers")
                    for title, excerpt, url, paper_id, published_date in related_papers:
                        display_paper_card(title, excerpt, url, paper_id, published_date)
                            
            except Exception as e:
                st.error(f"❌ An error occurred while searching for papers: {str(e)}")

if __name__ == "__main__":
    main()
