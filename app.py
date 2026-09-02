import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from sentence_transformers import util
from utils import inject_custom_css, load_data, load_embedder, get_corpus_embeddings

st.set_page_config(page_title="Myntra Discovery Engine", layout="wide")
inject_custom_css()

try:
    tagged, clustered, ranking = load_data()
    
    # Calculate sidebar metrics
    df_neg = tagged[tagged['hypothesis_tag'] != "general positive satisfaction with no specific complaint"]
    avg_confidence = df_neg['hypothesis_confidence'].mean()
    
    total_neg_in_tagged = len(df_neg)
    total_in_ranking = ranking['count'].sum() if not ranking.empty else 0
    
    if total_neg_in_tagged > 0:
        dropped_pct = ((total_neg_in_tagged - total_in_ranking) / total_neg_in_tagged) * 100
    else:
        dropped_pct = 0.0

    st.sidebar.markdown("### About")
    st.sidebar.write("The Myntra Discovery Engine analyzes unstructured social and review data to uncover the hidden friction points causing wishlisted items to drop off before purchase.")
    st.sidebar.divider()
    
    st.sidebar.metric("Avg. Classification Confidence", f"{avg_confidence:.2f}")
    st.sidebar.metric("Low-Confidence Rows Filtered", f"{dropped_pct:.1f}%")

    st.title("Myntra Discovery Engine - Customer Insights")
    
    # Top Banner
    st.markdown("""
    <div style="background-color: #e6e6e6; padding: 20px; border-radius: 8px; border-left: 5px solid #F5088B; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <h4 style="color: #282C3F; margin-top: 0;">Analysing 6,181 real reviews (5,014 Play Store · 465 Reddit · 702 App Store) to understand why wishlisted items don't convert to purchases.</h4>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Ask Why They Didn't Buy", "Opportunity Areas", "Discovery Findings", "How It Works"])
    
    with tab1:
        st.header("Ask Why They Didn't Buy")
        st.write("Search the underlying dataset for specific answers.")
        
        custom_query = st.text_input("Enter your custom question here:")
        
        st.write("**Or try one of these suggested questions:**")
        q1 = "Why do users hesitate before buying wishlisted items?"
        q2 = "What causes the most return/refund complaints?"
        q3 = "Do users trust Myntra's product authenticity?"
        q4 = "What delivery issues do users report most?"
        q5 = "What UX problems affect the wishlist feature?"
        
        b1, b2, b3, b4, b5 = st.columns(5)
        active_query = None
        
        if b1.button("Wishlist Hesitation", help=q1): active_query = q1
        if b2.button("Return Complaints", help=q2): active_query = q2
        if b3.button("Trust & Authenticity", help=q3): active_query = q3
        if b4.button("Delivery Issues", help=q4): active_query = q4
        if b5.button("Wishlist UX", help=q5): active_query = q5
        
        if st.button("Search Custom Question"):
            active_query = custom_query
            
        if active_query:
            if not active_query.strip():
                st.warning("Please enter a question.")
            else:
                st.info(f"**Query:** {active_query}")
                with st.spinner("Retrieving relevant reviews..."):
                    embedder = load_embedder()
                    texts = clustered['text'].astype(str).tolist()
                    corpus_embeddings = get_corpus_embeddings(embedder, texts)
                    
                    query_embedding = embedder.encode(active_query, convert_to_tensor=True)
                    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=8)[0]
                    
                    context_texts = []
                    for i, hit in enumerate(hits):
                        idx = hit['corpus_id']
                        row = clustered.iloc[idx]
                        context_texts.append(f"Review {i+1} (Source: {row['source']}): {row['text']}")
                        
                    context_str = "\\n\\n".join(context_texts)
                    
                    prompt = f"""
You are an expert product analyst. Please answer the following question using ONLY the provided reviews.
Cite which reviews you drew from (e.g., "According to Review 1...").

Question: {active_query}

Reviews:
{context_str}
"""
                
                with st.spinner("Synthesizing answer with Gemini..."):
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key)
                        
                        models_to_try = ["gemini-2.5-flash-lite", "gemini-3-flash", "gemini-3.6-flash"]
                        response = None
                        succeeded_model = None
                        last_error = None
                        
                        for model_name in models_to_try:
                            try:
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=300))
                                succeeded_model = model_name
                                break
                            except Exception as e:
                                last_error = e
                                continue
                        
                        if not response:
                            raise Exception(f"All fallback models failed. Last error: {last_error}")
                        
                        st.success(f"Answer Synthesized! (Model: `{succeeded_model}`)")
                        st.markdown(response.text)
                        
                        with st.expander("View Retrieved Source Reviews"):
                            for text in context_texts:
                                st.markdown(f"- {text}")
                    except KeyError:
                        st.error("Error: GEMINI_API_KEY not found in Streamlit secrets.")
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")

    with tab2:
        st.header("Opportunity Areas")
        st.write("These represent the top friction points causing users to hesitate, ranked by frequency.")
        
        # We map avg_negativity_intensity to color to highlight high-negativity bars using a magenta-to-orange scale
        fig_ranking = px.bar(ranking, x='count', y='hypothesis_tag', orientation='h',
                             color='avg_negativity_intensity',
                             color_continuous_scale=["#FF7A1A", "#F5088B"],
                             text='percentage_of_total',
                             title="Top Friction Points by Frequency")
        fig_ranking.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_ranking.update_traces(
            hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{text}%',
            texttemplate='%{text}%', 
            textposition='outside'
        )
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        st.divider()
        
        st.header("Cluster Explorer")
        st.write("Explore raw user feedback categorized by semantic clusters.")
        valid_clusters = sorted([c for c in clustered['cluster_id'].unique() if c != -1])
        selected_cluster = st.selectbox("Select a Cluster ID to explore:", valid_clusters)
        
        if selected_cluster is not None:
            cluster_data = clustered[clustered['cluster_id'] == selected_cluster]
            st.write(f"**Cluster {selected_cluster}** ({len(cluster_data)} rows)")
            st.dataframe(cluster_data[['source', 'text']].head(50), use_container_width=True)

    with tab3:
        st.header("Discovery Findings")
        
        st.subheader("Cross-Source Triangulation")
        st.write("This table shows the percentage of reviews from each source that fall into a given complaint category.")
        
        pivot_df = pd.crosstab(tagged['hypothesis_tag'], tagged['source'], normalize='columns') * 100
        pivot_df = pivot_df.round(1).astype(str) + '%'
        st.dataframe(pivot_df, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #fff0f4; padding: 20px; border-radius: 8px; border-left: 5px solid #F5088B; margin-top: 20px;">
            <h4 style="color: #282C3F; margin-top: 0;">Key Discovery Finding</h4>
            <p><em>[AI synthesis and initial analysis hypothesis to go here... Note: Requires primary research validation]</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 8px; border-left: 5px solid #FF7A1A; margin-top: 20px;">
            <h4 style="color: #282C3F; margin-top: 0;">Discovered Constraint: Wishlist Feature Limit</h4>
            <p>Through deep RAG queries against the corpus, we discovered a crucial structural constraint: users are explicitly complaining about a hard limit on the number of items they can add to their wishlist. This forces users to use the wishlist as a strict curation tool rather than a casual holding area, creating hidden friction before the checkout phase. (Pending qualitative interview confirmation)</p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("How It Works")
        
        st.markdown("""
        ### The 4-Stage Pipeline
        
        1. **Scraping**: We ingested raw feedback across multiple disparate platforms to capture a wide array of user intent.
        2. **Consolidation & Deduplication**: The data was aggressively cleaned to remove short, empty, or unhelpful spam, leaving us with a highly concentrated dataset of meaningful complaints.
        3. **Embedding & Clustering**:
           - **Model**: `Sentence-BERT` (`all-MiniLM-L6-v2`) mapped every text to a high-dimensional vector.
           - **Dimensionality Reduction**: `UMAP` compressed the embeddings.
           - **Clustering**: `HDBSCAN` automatically detected semantic groups of complaints without requiring predefined topics.
        4. **Hypothesis Classification**:
           - **Model**: `BART-large-MNLI` applied Zero-Shot Classification to tag every row against 8 specific wishlist-friction hypotheses.
           - **Synthesis**: A Gemini RAG pipeline (Retrieval-Augmented Generation) was built to allow direct natural language querying against the underlying data.
           
        ---
        
        ### Known Limitations
        
        - **Quora Data**: Scraping Quora yielded virtually no usable data (only 1 row remained before cleaning), so it was discarded.
        - **App Store Scraping**: Apple's App Store required multiple complex scraping approaches (like utilizing Apify) to bypass rate limits and gather the necessary 700+ rows.
        - **Silent Behaviors**: Direct evidence of *why* someone wishlists an item (but doesn't buy) is sparse. Wishlisting is often a "structurally silent" behavior—users rarely write a review saying "I wishlisted this but didn't buy." We had to infer hesitation by analyzing post-purchase regrets (returns, fake products, price drops) that cause a user to hesitate *the next time* they use the wishlist.
        """)

except Exception as e:
    st.error(f"Error loading data: {e}")
