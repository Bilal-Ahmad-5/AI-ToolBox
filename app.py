import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI ToolBox",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }
    .task-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-style: italic;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache models for performance
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_text_classification():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_ner_model():
    return pipeline("ner", grouped_entities=True)

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_translation_model():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

@st.cache_resource
def load_text_generation_model():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Helper functions
def download_button(data, filename, label):
    """Create download button for results"""
    if isinstance(data, dict):
        data = json.dumps(data, indent=2)
    elif isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    return href

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">💼 AI ToolBox</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Professional AI-Powered Tools for High-Demand Freelance Tasks")
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x80/667eea/ffffff?text=AI+ToolBox", use_container_width=True)
    st.sidebar.title("🎯 Task Categories")
    
    category = st.sidebar.selectbox(
        "Select Category:",
        ["🏠 Home", "📝 Text & Content Tasks", "💼 Business & Data Tasks", "🎨 Creative Tasks"]
    )
    
    # Home Page
    if category == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="task-card">
                <h3>📝 Text & Content</h3>
                <p>Sentiment analysis, NER, summarization, translation, and more</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="task-card">
                <h3>💼 Business & Data</h3>
                <p>Q&A, similarity detection, clustering, and analytics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="task-card">
                <h3>🎨 Creative Tasks</h3>
                <p>Content generation, writing assistance, and creative AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🌟 Why AI Task Master?")
        st.markdown("""
        - ✅ **Pre-trained Models**: No setup required, instant results
        - ✅ **Professional Quality**: Production-ready AI models
        - ✅ **Export Results**: Download as JSON or CSV
        - ✅ **Freelance Ready**: Perfect for Fiverr, Upwork, and client projects
        """)
    
    # Text & Content Tasks
    elif category == "📝 Text & Content Tasks":
        task = st.sidebar.radio(
            "Choose Task:",
            ["Sentiment Analysis", "Text Classification", "Named Entity Recognition", 
             "Text Summarization", "Translation", "SEO Keyword Extraction", "Content Generation"]
        )
        
        if task == "Sentiment Analysis":
            st.header("😊 Sentiment Analysis")
            st.markdown("Detect emotional tone in reviews, tweets, feedback, or any text.")
            
            sample_text = "I absolutely loved this product! It exceeded all my expectations and the customer service was amazing."
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter text to analyze:", height=150)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            if st.button("🔍 Analyze Sentiment"):
                if text_input:
                    with st.spinner("Analyzing sentiment..."):
                        model = load_sentiment_model()
                        result = model(text_input)[0]
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success(f"**Sentiment:** {result['label']}")
                        st.metric("Confidence Score", f"{result['score']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download option
                        result_json = json.dumps(result, indent=2)
                        st.download_button("📥 Download Result", result_json, "sentiment_result.json")
                        
                        with st.expander("💡 Freelance Tips"):
                            st.markdown("""
                            **How to sell this service:**
                            - Social media monitoring ($50-200/project)
                            - Product review analysis ($100-500/project)
                            - Brand reputation tracking ($200-1000/month)
                            - Customer feedback categorization ($150-400/project)
                            """)
                else:
                    st.warning("Please enter text to analyze.")
        
        elif task == "Text Classification":
            st.header("🏷️ Text Classification")
            st.markdown("Automatically categorize documents, emails, or support tickets.")
            
            sample_text = "I need help resetting my password. I've tried multiple times but haven't received the reset email."
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter text to classify:", height=150)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            if st.button("🔍 Classify Text"):
                if text_input:
                    with st.spinner("Classifying..."):
                        model = load_text_classification()
                        result = model(text_input)[0]
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success(f"**Category:** {result['label']}")
                        st.metric("Confidence", f"{result['score']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button("📥 Download Result", json.dumps(result, indent=2), "classification_result.json")
                else:
                    st.warning("Please enter text to classify.")
        
        elif task == "Named Entity Recognition":
            st.header("🔍 Named Entity Recognition (NER)")
            st.markdown("Extract names, organizations, locations, dates, and more from text.")
            
            sample_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976. The company is now worth over $2 trillion."
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter text for entity extraction:", height=150)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            if st.button("🔍 Extract Entities"):
                if text_input:
                    with st.spinner("Extracting entities..."):
                        model = load_ner_model()
                        entities = model(text_input)
                        
                        if entities:
                            df = pd.DataFrame(entities)
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.download_button("📥 Download Results", df.to_csv(index=False), "entities.csv")
                            
                            with st.expander("💡 Freelance Tips"):
                                st.markdown("""
                                **How to sell this service:**
                                - Document processing ($100-300/project)
                                - Contact extraction from resumes ($50-150/100 resumes)
                                - Legal document analysis ($200-800/project)
                                - Research paper entity extraction ($150-500/project)
                                """)
                        else:
                            st.info("No entities found.")
                else:
                    st.warning("Please enter text for entity extraction.")
        
        elif task == "Text Summarization":
            st.header("📄 Text Summarization")
            st.markdown("Condense long articles, reports, or documents into concise summaries.")
            
            sample_text = """Artificial intelligence is transforming the way we work and live. Machine learning algorithms can now 
            process vast amounts of data and make predictions with remarkable accuracy. Natural language processing enables 
            computers to understand and generate human language. Computer vision allows machines to interpret and analyze 
            visual information from the world. Deep learning, a subset of machine learning, has achieved breakthrough results 
            in image recognition, speech recognition, and game playing. AI is being applied across industries including healthcare, 
            finance, transportation, and entertainment. However, it also raises important ethical questions about privacy, 
            bias, and the future of work that society must address."""
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter text to summarize:", height=200)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            max_length = st.slider("Summary length (words):", 30, 150, 60)
            
            if st.button("📝 Summarize"):
                if text_input:
                    if len(text_input.split()) < 50:
                        st.warning("Text is too short to summarize. Please enter at least 50 words.")
                    else:
                        with st.spinner("Generating summary..."):
                            model = load_summarization_model()
                            summary = model(text_input, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.success("**Summary:**")
                            st.write(summary)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.download_button("📥 Download Summary", summary, "summary.txt")
                else:
                    st.warning("Please enter text to summarize.")
        
        elif task == "Translation":
            st.header("🌍 Translation")
            st.markdown("Translate text from multiple languages to English.")
            
            sample_text = "Bonjour, comment allez-vous aujourd'hui?"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter text to translate:", height=150)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            if st.button("🌍 Translate"):
                if text_input:
                    with st.spinner("Translating..."):
                        try:
                            model = load_translation_model()
                            translation = model(text_input)[0]['translation_text']
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.success("**Translation:**")
                            st.write(translation)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.download_button("📥 Download Translation", translation, "translation.txt")
                        except Exception as e:
                            st.error(f"Translation error: {str(e)}")
                else:
                    st.warning("Please enter text to translate.")
        
        elif task == "SEO Keyword Extraction":
            st.header("🔑 SEO Keyword Extraction")
            st.markdown("Extract key topics and themes from content for SEO optimization.")
            
            sample_text = """Digital marketing strategies for small businesses include social media marketing, 
            content marketing, email campaigns, and search engine optimization. Understanding your target audience 
            is crucial for effective marketing."""
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter content for keyword extraction:", height=150)
            with col2:
                if st.button("📝 Try Example"):
                    text_input = sample_text
                    st.rerun()
            
            if st.button("🔑 Extract Keywords"):
                if text_input:
                    with st.spinner("Extracting keywords..."):
                        # Simple keyword extraction using word frequency
                        words = text_input.lower().split()
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
                        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
                        word_freq = pd.Series(filtered_words).value_counts().head(10)
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success("**Top Keywords:**")
                        st.bar_chart(word_freq)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        df = pd.DataFrame({'Keyword': word_freq.index, 'Frequency': word_freq.values})
                        st.download_button("📥 Download Keywords", df.to_csv(index=False), "keywords.csv")
                else:
                    st.warning("Please enter content for keyword extraction.")
        
        elif task == "Content Generation":
            st.header("✍️ Content Generation")
            st.markdown("Generate creative content for social media, blogs, and marketing.")
            
            prompt = st.text_input("Enter a prompt or topic:", placeholder="Write a short blog intro about AI...")
            max_length = st.slider("Maximum length:", 50, 200, 100)
            
            if st.button("✨ Generate Content"):
                if prompt:
                    with st.spinner("Generating content..."):
                        model = load_text_generation_model()
                        generated = model(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success("**Generated Content:**")
                        st.write(generated)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button("📥 Download Content", generated, "generated_content.txt")
                        
                        with st.expander("💡 Freelance Tips"):
                            st.markdown("""
                            **How to sell this service:**
                            - Social media captions ($20-50/10 captions)
                            - Blog post writing ($100-500/post)
                            - Product descriptions ($50-200/20 descriptions)
                            - Email copy ($75-300/campaign)
                            """)
                else:
                    st.warning("Please enter a prompt.")
    
    # Business & Data Tasks
    elif category == "💼 Business & Data Tasks":
        task = st.sidebar.radio(
            "Choose Task:",
            ["Question Answering", "Text Similarity", "Document Summary Report", "Data Analytics (CSV)"]
        )
        
        if task == "Question Answering":
            st.header("❓ Question Answering")
            st.markdown("Extract answers from documents or knowledge bases.")
            
            context = st.text_area("Context/Document:", height=150, 
                placeholder="Paste the document or context here...")
            question = st.text_input("Question:", placeholder="What is this about?")
            
            if st.button("🔍 Find Answer"):
                if context and question:
                    with st.spinner("Finding answer..."):
                        model = load_qa_model()
                        result = model(question=question, context=context)
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success(f"**Answer:** {result['answer']}")
                        st.metric("Confidence", f"{result['score']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button("📥 Download Result", json.dumps(result, indent=2), "qa_result.json")
                else:
                    st.warning("Please provide both context and question.")
        
        elif task == "Text Similarity":
            st.header("🔗 Text Similarity Detection")
            st.markdown("Compare documents for similarity and detect duplicates.")
            
            text1 = st.text_area("Text 1:", height=100)
            text2 = st.text_area("Text 2:", height=100)
            
            if st.button("🔍 Calculate Similarity"):
                if text1 and text2:
                    with st.spinner("Calculating similarity..."):
                        model = load_sentence_transformer()
                        embeddings = model.encode([text1, text2])
                        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.metric("Similarity Score", f"{similarity:.2%}")
                        
                        if similarity > 0.8:
                            st.error("⚠️ Very similar - Possible duplicate")
                        elif similarity > 0.6:
                            st.warning("⚡ Moderately similar")
                        else:
                            st.success("✅ Different content")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        with st.expander("💡 Freelance Tips"):
                            st.markdown("""
                            **How to sell this service:**
                            - Plagiarism detection ($100-400/project)
                            - Duplicate content checking ($50-200/100 documents)
                            - Content uniqueness verification ($75-250/project)
                            """)
                else:
                    st.warning("Please enter both texts.")
        
        elif task == "Document Summary Report":
            st.header("📊 Document Summary Report")
            st.markdown("Generate comprehensive reports with summary and sentiment analysis.")
            
            text_input = st.text_area("Enter document text:", height=200)
            
            if st.button("📊 Generate Report"):
                if text_input:
                    with st.spinner("Generating report..."):
                        # Summarization
                        if len(text_input.split()) >= 50:
                            sum_model = load_summarization_model()
                            summary = sum_model(text_input, max_length=100, min_length=30)[0]['summary_text']
                        else:
                            summary = text_input
                        
                        # Sentiment
                        sent_model = load_sentiment_model()
                        sentiment = sent_model(text_input)[0]
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("📄 Summary")
                        st.write(summary)
                        
                        st.subheader("😊 Sentiment Analysis")
                        st.success(f"{sentiment['label']} (Confidence: {sentiment['score']:.2%})")
                        
                        st.subheader("📈 Statistics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Words", len(text_input.split()))
                        col2.metric("Characters", len(text_input))
                        col3.metric("Sentences", text_input.count('.') + text_input.count('!'))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        report = f"SUMMARY:\n{summary}\n\nSENTIMENT: {sentiment['label']}\n\nSTATS:\nWords: {len(text_input.split())}"
                        st.download_button("📥 Download Report", report, "document_report.txt")
                else:
                    st.warning("Please enter document text.")
        
        elif task == "Data Analytics (CSV)":
            st.header("📊 Data Analytics (CSV)")
            st.markdown("Upload a CSV for quick insights, regression, or clustering.")
            
            uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                st.subheader("📈 Basic Statistics")
                st.write(df.describe())
                
                analysis_type = st.selectbox("Choose analysis:", ["Clustering", "Correlation Matrix"])
                
                if analysis_type == "Clustering":
                    if st.button("🔍 Run Clustering"):
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            X = df[numeric_cols].dropna()
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            kmeans = KMeans(n_clusters=3, random_state=42)
                            clusters = kmeans.fit_predict(X_scaled)
                            
                            df_result = df.copy()
                            df_result['Cluster'] = clusters
                            
                            st.success("✅ Clustering complete!")
                            st.dataframe(df_result, use_container_width=True)
                            st.download_button("📥 Download Results", df_result.to_csv(index=False), "clustered_data.csv")
                        else:
                            st.warning("Need at least 2 numeric columns for clustering.")
                
                elif analysis_type == "Correlation Matrix":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        corr = numeric_df.corr()
                        st.write(corr)
                        st.download_button("📥 Download Correlation", corr.to_csv(), "correlation.csv")
                    else:
                        st.warning("No numeric columns found for correlation.")
    
    # Creative Tasks
    elif category == "🎨 Creative Tasks":
        task = st.sidebar.radio(
            "Choose Task:",
            ["AI Writing Assistant", "Creative Content Generator"]
        )
        
        if task == "AI Writing Assistant":
            st.header("✍️ AI Writing Assistant")
            st.markdown("Expand, rewrite, or improve your paragraphs.")
            
            text_input = st.text_area("Enter your draft:", height=150, 
                placeholder="Write a rough draft and I'll help improve it...")
            
            action = st.selectbox("Choose action:", ["Expand", "Rewrite", "Continue"])
            
            if st.button("✨ Process"):
                if text_input:
                    with st.spinner(f"{action}ing text..."):
                        model = load_text_generation_model()
                        
                        if action == "Expand":
                            prompt = f"Expand and elaborate on the following: {text_input}"
                        elif action == "Rewrite":
                            prompt = f"Rewrite the following in a better way: {text_input}"
                        else:
                            prompt = text_input
                        
                        result = model(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success(f"**{action}ed Version:**")
                        st.write(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button("📥 Download Result", result, "improved_text.txt")
                else:
                    st.warning("Please enter text to process.")
        
        elif task == "Creative Content Generator":
            st.header("🎨 Creative Content Generator")
            st.markdown("Generate creative content for various purposes.")
            
            content_type = st.selectbox("Content Type:", 
                ["Social Media Caption", "Blog Introduction", "Product Description", "Email Subject Line"])
            
            topic = st.text_input("Topic/Product:", placeholder="e.g., Eco-friendly water bottle")
            keywords = st.text_input("Keywords (optional):", placeholder="sustainable, BPA-free, durable")
            
            if st.button("🎨 Generate"):
                if topic:
                    with st.spinner("Creating content..."):
                        model = load_text_generation_model()
                        
                        prompts = {
                            "Social Media Caption": f"Write an engaging social media caption about {topic}:",
                            "Blog Introduction": f"Write a compelling blog introduction about {topic}:",
                            "Product Description": f"Write an attractive product description for {topic}:",
                            "Email Subject Line": f"Write a catchy email subject line about {topic}:"
                        }
                        
                        prompt = prompts[content_type]
                        if keywords:
                            prompt += f" Include keywords: {keywords}"
                        
                        result = model(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success(f"**Generated {content_type}:**")
                        st.write(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button("📥 Download", result, f"{content_type.lower().replace(' ', '_')}.txt")
                else:
                    st.warning("Please enter a topic.")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Built with ❤️ using Hugging Face + Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()