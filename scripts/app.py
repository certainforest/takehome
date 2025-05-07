import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import torch
import os
from pathlib import Path
import sys
from sentence_transformers import SentenceTransformer, util

# config
st.set_page_config(
    page_title = "Timeline",
    page_icon = "📅",
    layout = "wide"
)

# Load model once at app startup
@st.cache_resource
def load_model():
    try:
        # Use an appropriate model name here - replace with your actual model
        model = SentenceTransformer('joe32140/ModernBERT-base-msmarco')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# Load article data with caching
@st.cache_data
def load_article_data():
    try:
        # Get the base directory (where scripts folder is)
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # data paths 
        embeddings_path = base_dir / "data" / "article_embeddings.pt"
        metadata_path = base_dir / "data" / "article_metadata.parquet"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
        
        article_embeddings = torch.load(embeddings_path)
        article_metadata = pd.read_parquet(metadata_path)
        
        return article_embeddings, article_metadata
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def search_articles(query, _embeddings, metadata, _model, k = 5, threshold=0.3, sort_key='relevance_score'):
    try:
        # make sure query is string
        if not isinstance(query, str):
            query = str(query)
            
        # encode the query
        query_embedding = torch.tensor(_model.encode(query))
        
        # ensure embeddings is a tensor 
        if isinstance(_embeddings, list):
            embeddings_tensor = torch.stack(_embeddings)
        else:
            embeddings_tensor = _embeddings
            
        query_embedding = query_embedding.cpu()
        embeddings_tensor = embeddings_tensor.cpu()
            
        # calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
        
        # filter
        valid = similarities >= threshold
        if not valid.any():
            return pd.DataFrame()
        
        # get topk of valid results
        valid_indices = torch.where(valid)[0]
        filtered_similarities = similarities[valid_indices]
        
        # handle case where fewer available than requested
        k = min(k, len(filtered_similarities))
        if k == 0:
            return pd.DataFrame()
        
        # get top 
        top_k_values, top_k_indices = torch.topk(filtered_similarities, k=k)
        actual_indices = valid_indices[top_k_indices]
        
        result_indices = actual_indices.cpu().numpy()
        result_scores = top_k_values.cpu().numpy()
        
        # create results df 
        results_df = metadata.iloc[result_indices].copy()
        results_df['relevance_score'] = result_scores
        
        # reorder columns
        cols = ['relevance_score', 'headline', 'summary', 'timesTags', 'firstPublished']
        other_cols = [col for col in results_df.columns if col not in cols]
        results_df = results_df[cols + other_cols]
        
        # sort by key
        results_df = results_df.sort_values(by=sort_key, ascending=False)
        
        return results_df
    except Exception as e:
        st.error(f"Search error: {e}")
        return pd.DataFrame()

def main():
    st.title("Topic Timeline")
    
    model = load_model()
    article_embeddings, article_metadata = load_article_data()
    
    if model is None or article_embeddings is None or article_metadata is None:
        st.error("Failed to load necessary components. Please check logs for details.")
        return
    
    with st.sidebar:
        # st.sidebar.markdown("# Timeline", unsafe_allow_html=True)
        st.header("Search query")
        
        # use form
        with st.form(key="search_form"):
            search_query = st.text_input("")
            top_k = st.slider("# results", min_value=1, max_value=20, value=5)
            threshold = st.slider("similarity threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            
            # wait for "search" click 
            search_button = st.form_submit_button("Search")

        success_message_placeholder = st.empty()
            
        
    if search_button or search_query:
        if not search_query:
            st.warning("Please enter a search query.")
            return
            
        # search
        with st.spinner("Searching articles..."):
            results = search_articles(
                search_query,
                article_embeddings,
                article_metadata,
                model,
                k=top_k,
                threshold=threshold,
                sort_key='relevance_score'
            )
        
        if not results.empty:
            results['date'] = pd.to_datetime(results['firstPublished'])
            results['formatted_date'] = results['date'].dt.strftime('%B %d, %Y')
            results['year'] = results['date'].dt.year
            results['month'] = results['date'].dt.month
            with st.sidebar:
                success_message_placeholder.markdown(
                    f"""
                    <div id="temp-success-message" style="
                        background-color: #D4EDDA; 
                        color: #155724; 
                        padding: 8px 10px; 
                        border-radius: 4px; 
                        margin-bottom: 10px;
                        font-size: 14px;
                        transition: opacity 1.5s ease-out;
                    ">
                    Found {len(results)} relevant articles
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            # sort desc date 
            results_sorted = results.sort_values('date', ascending = False)
            
            current_year = None
            current_month = None
            
            for i, row in results_sorted.iterrows():
                year = row['year']
                month = row['date'].strftime('%B')
                
                # add year/month headers when they change
                if year != current_year:
                    st.markdown(f"## {year}")
                    current_year = year
                    current_month = None
                    
                if month != current_month:
                    # st.markdown(f"### {month}")
                    current_month = month
                
                # create a cards for each article 
                with st.container():
                    col1, col2 = st.columns([1, 6])
                    with col1:
                        st.markdown(f"<div style='font-weight:bold; font-size:16px; text-align:center;'>{row['date'].strftime('%b %d')}</div>", unsafe_allow_html=True)
                        # st.markdown(f"Score: {row['relevance_score']:.2f}") # not really useful in this context rn 
                    
                    with col2:
                        st.markdown(f"<h4 style='margin-top:0; margin-bottom:5px;'>{row['headline']}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:13px; color:#666; margin-bottom:8px; font-style:italic;'>{row['bylines']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:14px; margin-bottom:8px;'>{row['summary']}</div>", unsafe_allow_html=True)
                                                
                        if 'url' in row and row['url']:
                            # st.markdown(f"<div style='margin-bottom:8px;'>{[Read full article]({row['url']})</div>")
                            st.markdown(f"""
                                            <a href="{row['url']}" style="
                                            display: inline-block;
                                            font-size: 12px;
                                            background-color: #F47B4F;
                                            color: #FFFFFF;
                                            padding: 2px 8px;
                                            border-radius: 3px;
                                            text-decoration: none;
                                            transition: all 0.2s ease;
                                            font-weight: 500;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                            text-align: center;
                                            position: relative;">
                                            <span style="margin-right:6px;">→</span> Read article
                                        </a>
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                        )

                        # show tags as pills
                        if 'timesTags' in row and row['timesTags'] is not None:
                            tags = []
                            
                            # parse logic 
                            # np array 
                            if isinstance(row['timesTags'], np.ndarray):
                                tags = [str(tag) for tag in row['timesTags'] if tag]
                            # list 
                            elif isinstance(row['timesTags'], list):
                                tags = [str(tag) for tag in row['timesTags'] if tag]
                            # string 
                            elif isinstance(row['timesTags'], str):
                                if row['timesTags'].strip():
                                    tags = [tag.strip() for tag in row['timesTags'].split(',') if tag.strip()]

                            if tags:
                                top_3 = tags[:3]
                                # Enhanced styling for tags
                                tags_html = ' '.join([
                                    f'''<span style="
                                        background-color:#f8f9fa; 
                                        color: #495057;
                                        padding: 2px 10px; 
                                        border-radius: 16px; 
                                        margin-right: 6px; 
                                        margin-bottom: 6px;
                                        font-size: 0.75em; 
                                        display: inline-block;
                                        border: 1px solid #e9ecef;
                                        font-weight: normal;
                                        ">
                                        {tag}
                                    </span>''' 
                                    for tag in top_3
                                ])
                                st.markdown(f"<div style='margin-bottom:10px;'>{tags_html}</div>", unsafe_allow_html=True)
                    st.markdown("---")

        else:
            st.info("No results found. Try a different search query or lower the similarity threshold.")
    else:
        st.info("Enter a search query and click 'Search' to begin.")

if __name__ == "__main__":
    main()