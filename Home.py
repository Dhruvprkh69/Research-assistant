import streamlit as st

# Set the page config
st.set_page_config(
    page_title="Research AI Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        margin-top: 1em;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-title {
        color: #1f77b4;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        color: #2c3e50;
        font-size: 1rem;
        line-height: 1.5;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stSlider {
        margin-bottom: 1rem;
    }
    .welcome-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.title("🔬 Research AI Assistant")
    st.markdown("""
        ### Navigation
        Select a tool to get started with your research.
    """)
    
    page = st.radio("Go to", ["Home", "Understand Any Research Paper", "Generate Citations", "Search Similar Papers"])
    
    st.markdown("---")
    st.markdown("""
        ### About
        This AI-powered research assistant helps you:
        - Understand complex research papers
        - Generate accurate citations
        - Find related papers
        - Save time in your research process
    """)

# Home Page
if page == "Home":
    st.markdown('<div class="welcome-text">Welcome to Research AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your AI-powered assistant for academic research</div>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">📄 Understand Any Research Paper</div>
                <div class="feature-description">
                    Upload any research paper and get:
                    - Detailed summaries
                    - Key concepts explained
                    - Interactive Q&A
                    - Literature review style analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">📚 Generate Citations</div>
                <div class="feature-description">
                    Generate accurate citations in multiple formats:
                    - APA
                    - MLA
                    - Chicago
                    - IEEE
                    - Download citations as text
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔍 Search Similar Papers</div>
                <div class="feature-description">
                    Find related research papers:
                    - Semantic similarity search
                    - Category-based filtering
                    - Paper summaries
                    - Direct links to papers
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <h3>Ready to get started?</h3>
            <p>Select a tool from the sidebar to begin your research journey!</p>
        </div>
    """, unsafe_allow_html=True)

# Navigate to the "Understand Any Research Paper" page
elif page == "Understand Any Research Paper":
    st.write("🔍 Redirecting to **Understand Any Research Paper**...")
    import understand_paper  # Import the module
    understand_paper.main()  # Call the main function from understand_paper.py

# Navigate to the "Generate Citations" page
elif page == "Generate Citations":
    st.write("📚 Redirecting to **Generate Citations**...")
    import citation_generator  # Import the module
    citation_generator.main()  # Call the main function from citation_generator.py

# Navigate to the "Search Similar Papers" page
elif page == "Search Similar Papers":
    st.write("📄 Redirecting to **Search Similar Papers**...")
    import search_similar_papers  # Import the module
    search_similar_papers.main()  # Call the main function from search_similar_papers.py

