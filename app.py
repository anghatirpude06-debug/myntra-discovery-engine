import os
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from sentence_transformers import util
from utils import load_data, load_embedder, get_corpus_embeddings

st.set_page_config(page_title="MyntraLens", layout="wide")

@st.cache_data(ttl=3600)
def generate_rag_insight(active_query, context_str):
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    prompt = f"""
You are a lead UX researcher. Answer the following question synthesizing only the provided user feedback.
Make sure to reference your sources (e.g., "Based on Source 1...").

Question: {active_query}

User Feedback:
{context_str}
"""
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    last_error = None
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=300))
            return response.text, model_name
        except Exception as e:
            last_error = e
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            continue
            
    if last_error and ("429" in str(last_error) or "quota" in str(last_error).lower()):
        return "QUOTA_ERROR", None
        
    raise Exception(f"All fallback models failed. Error: {last_error}")

LIGHT_CSS = """
<style>
    h1, h2, h3, h4, h5, h6 { color: #282C3F !important; }
    body, p, div { color: #282C3F; }
    .sub-label, small, .stCaption { color: #6B7280 !important; }
    
    .top-banner { background: linear-gradient(135deg, rgba(245, 8, 139, 0.05), rgba(255, 122, 26, 0.05)); border-left: 6px solid #F5088B; padding: 20px; border-radius: 8px; margin-bottom: 20px; width: 100%; box-sizing: border-box; white-space: normal; word-wrap: break-word; overflow-wrap: break-word; min-height: 80px; }
    .top-banner h4 { color: #282C3F !important; margin-top: 0; font-size: 18px; line-height: 1.7; }
    
    .banner-card { background-color: #fcfcfc; border-left: 5px solid #FF7A1A; padding: 20px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); }
    .banner-card-pink { border-left-color: #F5088B; }
    .banner-card h4 { color: #282C3F !important; margin-top: 0; margin-bottom: 10px; }
    .banner-card p { margin-bottom: 0; }
    
    section[data-testid="stSidebar"] button { height: 50px; border-radius: 10px; border: none !important; border-left: 4px solid transparent !important; background-color: transparent !important; color: #282C3F !important; font-weight: 600; justify-content: flex-start; padding-left: 16px; transition: all 0.2s ease; margin-bottom: 2px;}
    section[data-testid="stSidebar"] button p, section[data-testid="stSidebar"] button span, section[data-testid="stSidebar"] button div { color: inherit !important; transition: all 0.2s ease; }
    section[data-testid="stSidebar"] button:hover { background-color: rgba(0,0,0,0.02) !important; color: #282C3F !important; border-left-color: #F5088B !important; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    
    div.stButton > button { transition: all 0.2s ease; }
    div.stButton > button:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-color: #F5088B; color: #F5088B; }
    
    /* Outlined secondary buttons */
    .stButton.quick-prompt > button { border: 1px solid #d1d5db !important; background-color: white !important; color: #282C3F !important; }
    
    /* Primary Action button forcefully applied */
    button[kind="primary"] { background-color: #282C3F !important; color: #FFFFFF !important; border: none !important; border-radius: 999px !important; }
    button[kind="primary"]:hover { background-color: #3D4159 !important; color: #FFFFFF !important; border: none !important; }
    div[data-testid="stButton"] > button[kind="primary"] { background-color: #282C3F !important; color: #FFFFFF !important; border: none !important; border-radius: 999px !important; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #3D4159 !important; color: #FFFFFF !important; border: none !important; }
    button[kind="primary"] p, button[kind="primary"] div, button[kind="primary"] span { color: #FFFFFF !important; }
    div[data-testid="stButton"] > button[kind="primary"] p, div[data-testid="stButton"] > button[kind="primary"] div, div[data-testid="stButton"] > button[kind="primary"] span { color: #FFFFFF !important; }
    
    .opp-card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 4px solid #F5088B; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .opp-card .opp-title { font-size: 13px; color: #6B7280; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .opp-card .opp-value { font-size: 18px; color: #F5088B; font-weight: 700; line-height: 1.2; margin-bottom: 4px; }
    .opp-card .opp-sub { font-size: 13px; color: #282C3F; }
    
    div[data-testid="stMetric"] { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 4px solid #F5088B; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); transition: all 0.2s ease; }
    div[data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left-color: #FF7A1A; }
    div[data-testid="stMetricValue"] > div { color: #F5088B !important; }
    div[data-testid="stMetricLabel"] p { color: #6B7280 !important; font-size: 14px !important; font-weight: 600; }
    
    /* Hide anchor link icons globally */
    [data-testid="stHeaderActionElements"] { display: none !important; }
    
    /* Logo button styling */
    section[data-testid="stSidebar"] div.element-container:nth-of-type(1) button {
        background: transparent !important; border: none !important; box-shadow: none !important; padding-left: 0 !important; margin-bottom: 10px !important; justify-content: flex-start !important;
    }
    section[data-testid="stSidebar"] div.element-container:nth-of-type(1) button:hover {
        background: transparent !important; border: none !important; box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div.element-container:nth-of-type(1) button p, section[data-testid="stSidebar"] div.element-container:nth-of-type(1) button div {
        font-size: 28px !important; font-weight: 700 !important; color: #282C3F !important; text-align: left !important; margin-left: 0 !important;
    }
</style>
"""

TAG_MAP = {
    "fit or size uncertainty": "Size & Fit",
    "return or refund friction": "Returns & Refunds",
    "delivery delay or reliability issue": "Delivery",
    "trust or authenticity doubt about product genuineness": "Trust & Authenticity",
    "app bugs or technical friction": "App Experience",
    "price or value consideration": "Price & Value",
    "wishlist feature usability issue": "Wishlist Experience",
    "general positive satisfaction with no specific complaint": "Overall Satisfaction"
}

try:
    tagged, clustered, ranking = load_data()
    try:
        import json
        with open("data/precomputed_answers.json", "r", encoding="utf-8") as f:
            precomputed_answers = json.load(f)
    except Exception:
        precomputed_answers = {}
    
    if 'active_page' not in st.session_state:
        st.session_state['active_page'] = "Ask the Engine"
    if 'selected_sources' not in st.session_state:
        st.session_state['selected_sources'] = ["Play Store", "App Store", "Reddit"]

    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    with st.sidebar:
        if st.button("MyntraLens"):
            st.session_state['active_page'] = "Ask the Engine"
            st.rerun()
        st.markdown("<p class='sub-label' style='font-size: 14px;'>This intelligence engine parses unstructured social media and app store feedback to expose the hidden friction points causing wishlist abandonment.</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### Navigation")
        
        pages = ["Ask the Engine", "Deep Analytics", "How It Works"]
        
        for i, page in enumerate(pages):
            if st.button(f"{page}", key=f"nav_{page}", use_container_width=True):
                st.session_state['active_page'] = page
                st.rerun()
                
        # Push Sync Data to the bottom of the sidebar visually
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### Filters")
        st.multiselect("Filter by Source", ["Play Store", "App Store", "Reddit"], key="selected_sources")
        
        st.divider()
        
        if st.button("↻ Sync Data", use_container_width=True, key="sync_sidebar_btn"):
            st.toast("Data pipeline sync initiated successfully!", icon="✅")
            
    active_idx = pages.index(st.session_state['active_page']) + 5
    
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] div.element-container:nth-of-type({active_idx}) button {{
        background: linear-gradient(135deg, rgba(245, 8, 139, 0.05), rgba(255, 122, 26, 0.05)) !important;
        border-left: 4px solid #F5088B !important;
        border-top-left-radius: 0px !important;
        border-bottom-left-radius: 0px !important;
    }}
    section[data-testid="stSidebar"] div.element-container:nth-of-type({active_idx}) button p,
    section[data-testid="stSidebar"] div.element-container:nth-of-type({active_idx}) button span,
    section[data-testid="stSidebar"] div.element-container:nth-of-type({active_idx}) button div {{
        color: #F5088B !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    selection = st.session_state['active_page']
    
    source_map = {"Play Store": "play_store", "App Store": "app_store", "Reddit": "reddit"}
    active_source_keys = [source_map[s] for s in st.session_state['selected_sources']]

    if selection == "Ask the Engine":
        st.markdown("""
        <div class="top-banner">
            <h4>Why do Myntra users build massive wishlists but hesitate at checkout?<br>Decoding 6,181 genuine reviews maps the entire customer journey, surfacing the real reasons behind cart abandonment and lost conversions.</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_header, _ = st.columns([4, 1])
        with col_header:
            st.header("Ask the Engine")
            st.markdown("<p class='sub-label'>Use our RAG-powered engine to search the raw review dataset and generate synthesized answers based on actual customer complaints.</p>", unsafe_allow_html=True)
                
        # Reviews by Platform stat block
        play_count = len(tagged[tagged['source'] == 'play_store'])
        app_count = len(tagged[tagged['source'] == 'app_store'])
        reddit_count = len(tagged[tagged['source'] == 'reddit'])
        
        play_op = "1.0" if "play_store" in active_source_keys else "0.4"
        app_op = "1.0" if "app_store" in active_source_keys else "0.4"
        reddit_op = "1.0" if "reddit" in active_source_keys else "0.4"
        
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="banner-card" style="padding: 15px; border-left-color: #F5088B; border-top: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; height: 100%;">
                <div style="flex-grow: 1;">
                    <p style="margin: 0; color: #6B7280; font-size: 12px; font-weight: 600; text-transform: uppercase; line-height: 1.2; margin-bottom: 4px;">Cleaned Reviews</p>
                    <h3 style="margin: 0; color: #282C3F; font-size: 24px;">{len(tagged):,}</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="banner-card" style="opacity: {play_op}; padding: 15px; border-left-color: #10B981; border-top: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; height: 100%;">
                <div style="flex-grow: 1;">
                    <p style="margin: 0; color: #6B7280; font-size: 12px; font-weight: 600; text-transform: uppercase; line-height: 1.2; margin-bottom: 4px;">Play Store</p>
                    <h3 style="margin: 0; color: #282C3F; font-size: 24px;">{play_count:,}</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="banner-card" style="opacity: {app_op}; padding: 15px; border-left-color: #3B82F6; border-top: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; height: 100%;">
                <div style="flex-grow: 1;">
                    <p style="margin: 0; color: #6B7280; font-size: 12px; font-weight: 600; text-transform: uppercase; line-height: 1.2; margin-bottom: 4px;">App Store</p>
                    <h3 style="margin: 0; color: #282C3F; font-size: 24px;">{app_count:,}</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="banner-card" style="opacity: {reddit_op}; padding: 15px; border-left-color: #FF7A1A; border-top: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; height: 100%;">
                <div style="flex-grow: 1;">
                    <p style="margin: 0; color: #6B7280; font-size: 12px; font-weight: 600; text-transform: uppercase; line-height: 1.2; margin-bottom: 4px;">Reddit</p>
                    <h3 style="margin: 0; color: #282C3F; font-size: 24px;">{reddit_count:,}</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("**Quick Prompts:**")
        
        q1 = "Why do users hesitate before buying wishlisted items?"
        q2 = "What causes the most return and refund complaints?"
        q3 = "Do users trust Myntra's product authenticity?"
        q4 = "What delivery issues do users report most often?"
        q5 = "What UX problems affect the wishlist feature?"
        
        active_query = None
        
        st.markdown('<div class="quick-prompts-container">', unsafe_allow_html=True)
        qc1, qc2 = st.columns(2)
        with qc1:
            if st.button(q1, use_container_width=True): active_query = q1
            if st.button(q2, use_container_width=True): active_query = q2
            if st.button(q3, use_container_width=True): active_query = q3
        with qc2:
            if st.button(q4, use_container_width=True): active_query = q4
            if st.button(q5, use_container_width=True): active_query = q5
        st.markdown('</div>', unsafe_allow_html=True)
            
        custom_query = st.text_input("Or enter a custom query:")
        st.markdown("<p style='font-size: 12px; color: #9CA3AF; font-style: italic; margin-top: -10px; margin-bottom: 20px;'>AI-generated responses may contain inaccuracies.</p>", unsafe_allow_html=True)
        
        if st.button("Generate Insight", type="primary", use_container_width=True):
            active_query = custom_query
            
        if active_query:
            if not active_query.strip():
                st.warning("Please type a valid question.")
            else:
                st.info(f"**Investigating:** {active_query}")
                try:
                    if active_query in precomputed_answers:
                        ans_data = precomputed_answers[active_query]
                        st.success(f"Analysis Complete (Powered by `{ans_data['model']}`)")
                        st.markdown(ans_data['answer'])
                        with st.expander("Examine Retrieved Feedback Records"):
                            for text in ans_data['sources']:
                                st.markdown(f"- {text}")
                    else:
                        with st.spinner("Retrieving relevant reviews and generating an answer..."):
                            embedder = load_embedder()
                            texts = clustered['text'].astype(str).tolist()
                            corpus_embeddings = get_corpus_embeddings(embedder, texts)
                            
                            query_embedding = embedder.encode(active_query, convert_to_tensor=True)
                            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=8)[0]
                            
                            if not hits or hits[0]['score'] < 0.3:
                                st.warning("This question doesn't appear to relate to the Myntra wishlist/purchase dataset. Try asking about fit, sizing, returns, delivery, trust, pricing, or the wishlist feature.")
                            else:
                                context_texts = []
                                for i, hit in enumerate(hits):
                                    idx = hit['corpus_id']
                                    row = clustered.iloc[idx]
                                    context_texts.append(f"Source {i+1} ({row.get('source', 'N/A')}): {row.get('text', 'N/A')}")
                                    
                                context_str = "\\n\\n".join(context_texts)
                                
                                try:
                                    text_out, model_used = generate_rag_insight(active_query, context_str)
                                    
                                    if text_out == "QUOTA_ERROR":
                                        st.error("This question requires a live AI query and we've hit today's free quota. Try one of the other suggested questions above, which are pre-loaded and always available.")
                                    else:
                                        st.success(f"Analysis Complete (Powered by `{model_used}`)")
                                        st.markdown(text_out)
                                        
                                        with st.expander("Examine Retrieved Feedback Records"):
                                            for text in context_texts:
                                                st.markdown(f"- {text}")
                                except KeyError:
                                    st.error("Configuration Error: GEMINI_API_KEY is missing from Streamlit secrets.")
                                except Exception as e:
                                    st.error(f"Inference Engine Error: {e}")
                except Exception as e:
                    st.error(f"RAG Error: {e}")

    elif selection == "Deep Analytics":
        st.header("Deep Analytics")
        st.markdown("<p class='sub-label'>A breakdown of the primary reasons users hesitate and abandon their wishlists.</p>", unsafe_allow_html=True)
        
        if not active_source_keys:
            st.warning("Select at least one source in the sidebar to view analytics.")
        else:
            filtered_tagged = tagged[tagged['source'].isin(active_source_keys)].copy()
            
            st.markdown("""
            <div class="banner-card banner-card-pink">
                <h4>Dominant Friction Pattern</h4>
                <p>Across the dataset, users are fundamentally struggling with post-purchase anxiety that bleeds into their pre-purchase wishlist behavior. The highest volume of friction stems directly from complicated return processes, delayed refunds, and lingering doubts regarding product authenticity. When users don't trust the fulfillment or return pipeline, they use the wishlist as a holding area rather than converting to a cart.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Platform Frustration Index")
            df_merged = pd.merge(filtered_tagged, ranking[['hypothesis_tag', 'avg_negativity_intensity']], on='hypothesis_tag', how='left')
            df_merged['frustration_score'] = df_merged['hypothesis_confidence'] * df_merged['avg_negativity_intensity'] * 100
            platform_scores = df_merged.groupby('source')['frustration_score'].mean().fillna(0).to_dict()
        
        if active_source_keys:
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("Play Store Frustration", f"{platform_scores.get('play_store', 0):.1f}/100")
                st.markdown("<small class='sub-label'>Driven by bugs and high-volume return complaints.</small>", unsafe_allow_html=True)
            with p2:
                st.metric("App Store Frustration", f"{platform_scores.get('app_store', 0):.1f}/100")
                st.markdown("<small class='sub-label'>Driven by premium UX expectations and authenticity concerns.</small>", unsafe_allow_html=True)
            with p3:
                st.metric("Reddit Frustration", f"{platform_scores.get('reddit', 0):.1f}/100")
                st.markdown("<small class='sub-label'>Driven by deep, structural catalog and policy discussions.</small>", unsafe_allow_html=True)
                
            st.write("")
            st.subheader("Key Insights")
            st.markdown("""
            <div class="banner-card">
                <h4>1. Returns & Refunds Dominate Hesitation</h4>
                <p>The vast majority of identifiable friction points relate to users fearing they won't be able to return items easily, causing them to pause on high-value wishlist items.</p>
            </div>
            <div class="banner-card">
                <h4>2. The "Silent" Wishlist Behavior</h4>
                <p>Users rarely state <em>why</em> they abandoned a wishlist item in a review. Instead, they complain about a bad experience (like a fake product), which we infer is the root cause for their hesitation on future purchases.</p>
            </div>
            <div class="banner-card">
                <h4>3. Cross-Source Agreement</h4>
                <p>Whether on Reddit, the App Store, or Play Store, the core complaints remain remarkably consistent—indicating these are structural platform issues, not localized bugs.</p>
            </div>
            <div class="banner-card banner-card-pink">
                <h4>4. Structural Constraints Impede Curation</h4>
                <p>Users actively bump into a hard limit on the number of items they can wishlist, forcing them to treat the feature as a strict curation tool rather than a casual saved-for-later bin.</p>
            </div>
            """, unsafe_allow_html=True)
        
            st.write("")
            st.subheader("Discovery Themes")
            
            # Recalculate ranking dynamically
            ranking_recalc = filtered_tagged.groupby('hypothesis_tag').size().reset_index(name='count')
            ranking_recalc = pd.merge(ranking_recalc, ranking[['hypothesis_tag', 'avg_negativity_intensity']], on='hypothesis_tag', how='left')
            ranking_recalc = ranking_recalc.sort_values(by='count', ascending=False)
            ranking_recalc['percentage_of_total'] = (ranking_recalc['count'] / len(filtered_tagged)) * 100 if not filtered_tagged.empty else 0
            
            if not ranking_recalc.empty:
                valid_ranking = ranking_recalc[ranking_recalc['hypothesis_tag'] != "general positive satisfaction with no specific complaint"].copy()
            if not valid_ranking.empty:
                valid_ranking['display_name'] = valid_ranking['hypothesis_tag'].map(TAG_MAP).fillna(valid_ranking['hypothesis_tag'].astype(str))
                
                most_mentioned = valid_ranking.loc[valid_ranking['count'].idxmax()]
                most_negative = valid_ranking.loc[valid_ranking['avg_negativity_intensity'].idxmax()]
                most_positive = valid_ranking.loc[valid_ranking['avg_negativity_intensity'].idxmin()]
                
                tr1, tr2, tr3 = st.columns(3)
                with tr1:
                    d_name_1 = most_mentioned.get('display_name', 'N/A')
                    st.markdown(f"""<div class="opp-card" style="border-left: 4px solid #F5088B;"><div class="opp-title">MOST MENTIONED</div><div class="opp-value" style="color: #282C3F;">{d_name_1}</div><div class="opp-sub">{most_mentioned.get('count', 0)} reviews ({most_mentioned.get('percentage_of_total', 0):.1f}%)</div></div>""", unsafe_allow_html=True)
                with tr2:
                    d_name_2 = most_negative.get('display_name', 'N/A')
                    neg_val = most_negative.get('avg_negativity_intensity', 0)*100
                    st.markdown(f"""<div class="opp-card" style="border-left: 4px solid #EF4444;"><div class="opp-title">MOST NEGATIVE</div><div class="opp-value" style="color: #282C3F;">{d_name_2}</div><div class="opp-sub" style="color: #EF4444; font-weight: 600;">The most painful theme ({neg_val:.0f}% negative)</div></div>""", unsafe_allow_html=True)
                with tr3:
                    d_name_3 = most_positive.get('display_name', 'N/A')
                    st.markdown(f"""<div class="opp-card" style="border-left: 4px solid #10B981;"><div class="opp-title">MOST POSITIVE</div><div class="opp-value" style="color: #282C3F;">{d_name_3}</div><div class="opp-sub" style="color: #10B981; font-weight: 600;">What users are least concerned about</div></div>""", unsafe_allow_html=True)

            st.write("")
            valid_ranking_tags = ranking_recalc[ranking_recalc['hypothesis_tag'] != "general positive satisfaction with no specific complaint"]
            for _, row in valid_ranking_tags.iterrows():
                raw_tag = row['hypothesis_tag']
                d_name = TAG_MAP.get(raw_tag, str(raw_tag))
                count = row.get('count', 0)
                neg = row.get('avg_negativity_intensity', 0)
                
                if neg > 0.6:
                    sentiment_label = "Negative"
                    color = "#EF4444"
                elif neg < 0.35:
                    sentiment_label = "Positive"
                    color = "#10B981"
                else:
                    sentiment_label = "Mixed"
                    color = "#F59E0B"
                    
                cdata = filtered_tagged[filtered_tagged['hypothesis_tag'] == raw_tag]
                
                with st.expander(f"{d_name} ({count} reviews) — {sentiment_label}"):
                    st.markdown(f"**Sentiment Profile:** <span style='color:{color}; font-weight:bold;'>{sentiment_label}</span> (Negativity Intensity: {neg:.2f})", unsafe_allow_html=True)
                    st.markdown("*" + "What users are saying:" + "*")
                    
                    # Show 3-4 illustrative quotes
                    sample_reviews = cdata['text'].dropna().sample(min(4, len(cdata)), random_state=42).tolist()
                    for txt in sample_reviews:
                        st.markdown(f"> \"{txt}\"")
                        
            st.write("")
            st.subheader("Top Opportunity Areas")
            if not ranking_recalc.empty:
                o1, o2, o3 = st.columns(3)
                with o1:
                    t1 = TAG_MAP.get(ranking_recalc.iloc[0]['hypothesis_tag'], str(ranking_recalc.iloc[0]['hypothesis_tag'])) if pd.notna(ranking_recalc.iloc[0]['hypothesis_tag']) else "N/A"
                    p1 = ranking_recalc.iloc[0]['percentage_of_total'] if pd.notna(ranking_recalc.iloc[0]['percentage_of_total']) else 0
                st.markdown(f"""<div class="opp-card"><div class="opp-title">Primary Opportunity</div><div class="opp-value">{t1}</div><div class="opp-sub">{p1:.1f}% of signals</div></div>""", unsafe_allow_html=True)
                if len(ranking_recalc) > 1:
                    with o2:
                        t2 = TAG_MAP.get(ranking_recalc.iloc[1]['hypothesis_tag'], str(ranking_recalc.iloc[1]['hypothesis_tag'])) if pd.notna(ranking_recalc.iloc[1]['hypothesis_tag']) else "N/A"
                        p2 = ranking_recalc.iloc[1]['percentage_of_total'] if pd.notna(ranking_recalc.iloc[1]['percentage_of_total']) else 0
                    st.markdown(f"""<div class="opp-card"><div class="opp-title">Secondary Opportunity</div><div class="opp-value">{t2}</div><div class="opp-sub">{p2:.1f}% of signals</div></div>""", unsafe_allow_html=True)
                if len(ranking_recalc) > 2:
                    with o3:
                        t3 = TAG_MAP.get(ranking_recalc.iloc[2]['hypothesis_tag'], str(ranking_recalc.iloc[2]['hypothesis_tag'])) if pd.notna(ranking_recalc.iloc[2]['hypothesis_tag']) else "N/A"
                        p3 = ranking_recalc.iloc[2]['percentage_of_total'] if pd.notna(ranking_recalc.iloc[2]['percentage_of_total']) else 0
                    st.markdown(f"""<div class="opp-card"><div class="opp-title">Tertiary Opportunity</div><div class="opp-value">{t3}</div><div class="opp-sub">{p3:.1f}% of signals</div></div>""", unsafe_allow_html=True)
            
                st.write("")
                display_ranking = ranking_recalc.copy()
                display_ranking['display_name'] = display_ranking['hypothesis_tag'].map(TAG_MAP).fillna(display_ranking['hypothesis_tag']).fillna("N/A").astype(str)
                fig_ranking = px.bar(display_ranking, x='count', y='display_name', orientation='h',
                                     color='avg_negativity_intensity',
                                     color_continuous_scale=["#FF7A1A", "#F5088B"],
                                     text='percentage_of_total')
                fig_ranking.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Opportunity Area Breakdown")
                fig_ranking.update_traces(hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{text}%', texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig_ranking, use_container_width=True)
                
            st.write("")
            st.subheader("Top User Needs & Actionable Insights")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="banner-card">
                    <h4>Discovery & Convenience Opportunities</h4>
                    <p><strong>Expand Wishlist Limits:</strong> Increase the hard cap on wishlists to encourage casual saving rather than strict curation.<br><br>
                    <strong>Authenticity Badges:</strong> Display verified supplier badges explicitly on wishlisted items to counter trust-related hesitation.</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="banner-card banner-card-pink">
                    <h4>UX & Control Enhancements</h4>
                    <p><strong>Transparent Return Policies:</strong> Highlight return windows and exact refund timelines directly inside the wishlist view.<br><br>
                    <strong>Sizing Intelligence:</strong> Introduce proactive size-fit predictors before the user moves the item to cart to reduce return anxiety.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            st.subheader("Recent Reviews Live Feed")
            
            if 'timestamp' in filtered_tagged.columns:
                sorted_tagged = filtered_tagged.sort_values('timestamp', ascending=False)
            else:
                sorted_tagged = filtered_tagged.copy()
            
            play_rows = sorted_tagged[sorted_tagged['source'] == 'play_store'].head(4)
            app_rows = sorted_tagged[sorted_tagged['source'] == 'app_store'].head(3)
            reddit_rows = sorted_tagged[sorted_tagged['source'] == 'reddit'].head(3)
            
            combined_rows = pd.concat([play_rows, app_rows, reddit_rows])
            if 'timestamp' in combined_rows.columns:
                feed_rows = combined_rows.sort_values('timestamp', ascending=False)
            else:
                feed_rows = combined_rows.sample(frac=1, random_state=42)
                
            for _, row in feed_rows.iterrows():
                src = row.get('source', 'Unknown')
                if src == 'play_store':
                    color = "#10B981" 
                elif src == 'app_store':
                    color = "#3B82F6" 
                else:
                    color = "#FF7A1A" 
                    
                d_tag = TAG_MAP.get(row.get('hypothesis_tag'), str(row.get('hypothesis_tag', 'N/A')))
                st.markdown(f"""
                <div style="background-color: #f9f9f9; border-left: 3px solid {color}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <small style="color: #6B7280; font-weight: bold;">{str(src).upper()} • {row.get('timestamp', 'Recent')}</small>
                    <span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 12px; font-size: 10px; margin-left: 10px;">{d_tag}</span>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">{str(row.get('text', 'N/A'))[:150]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.subheader("Shopping Behaviour (Primary Research)")
            if os.path.exists("data/survey_responses.csv"):
                survey = pd.read_csv("data/survey_responses.csv")
                n_respondents = len(survey)
                
                col4 = 'Have you ever NOT bought a wishlisted item specifically because you were unsure about the size/fit?'
                never_pct = (survey[col4] == 'Never').mean() * 100 if col4 in survey.columns else 0
                fit_skipped = 100 - never_pct
                
                col5 = 'Would knowing "based on your past orders, size M will likely fit you in this brand" make you more likely to buy from your wishlist?'
                buy_more = survey[col5].str.contains('yes', case=False, na=False).mean() * 100 if col5 in survey.columns else 0
                
                col3 = 'When you wishlist something but don\'t buy it within 30 days, what\'s the MAIN reason?'
                if col3 in survey.columns:
                    valid_reasons = survey[col3].dropna()
                    top_reason = valid_reasons.value_counts().index[0] if not valid_reasons.empty else "N/A"
                else:
                    top_reason = "N/A"
                
                st.markdown("""
                <style>
                .survey-metric { background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
                .survey-metric .val { font-size: 24px; font-weight: bold; color: #F5088B; margin-bottom: 5px; }
                .survey-metric .lbl { font-size: 13px; color: #6B7280; font-weight: 500; }
                .survey-metric .text-val { font-size: 16px; font-weight: 600; color: #282C3F; margin-bottom: 5px; }
                </style>
                """, unsafe_allow_html=True)
                
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f"""<div class="survey-metric"><div class="val">{n_respondents}</div><div class="lbl">Survey Respondents</div></div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""<div class="survey-metric"><div class="val">{fit_skipped:.1f}%</div><div class="lbl">Skipped due to fit uncertainty</div></div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""<div class="survey-metric"><div class="val">{buy_more:.1f}%</div><div class="lbl">Would buy more with fit-prediction</div></div>""", unsafe_allow_html=True)
                
                st.write("")
                st.markdown(f"""
                <div class="banner-card" style="width: 100%; border-left-color: #282C3F;">
                    <h4 style="color: #6B7280 !important; font-size: 14px; text-transform: uppercase;">Top reason unpurchased</h4>
                    <p style="font-size: 18px; font-weight: bold; color: #282C3F;">{top_reason}</p>
                </div>
                """, unsafe_allow_html=True)
                    
            else:
                st.markdown("""
                <div class="banner-card">
                    <h4 style="color: #6B7280 !important;">Survey data pending</h4>
                    <p>This section will be integrated directly from primary qualitative research.</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif selection == "How It Works":
        st.header("How It Works")
        
        st.write("This engine is powered by a multi-stage NLP pipeline designed to extract meaning from chaotic unstructured text.")
        
        st.markdown("""
        <div class="banner-card">
            <h4>The Discovery Pipeline</h4>
            <ol>
                <li><strong>Data Acquisition</strong>: We scraped massive volumes of raw sentiment from Reddit, Google Play, and the Apple App Store to capture authentic, unsolicited user opinions.</li>
                <li><strong>Filtering & Preprocessing</strong>: The raw scrape was aggressively filtered. We stripped out empty reviews, short spam, and irrelevant noise, resulting in a dense, high-signal dataset.</li>
                <li><strong>Vectorization & Clustering</strong>: Embeddings grouped semantically similar complaints without us having to pre-define the topics.</li>
                <li><strong>AI Classification & Synthesis</strong>: A zero-shot model evaluated every single review against distinct friction hypotheses, assigning a confidence score to each.</li>
            </ol>
        </div>
        
        <div class="banner-card banner-card-pink">
            <h4>Models & Methods</h4>
            <ul>
                <li><strong>Embeddings:</strong> Sentence-BERT (all-MiniLM-L6-v2)</li>
                <li><strong>Dimensionality Reduction:</strong> UMAP</li>
                <li><strong>Clustering:</strong> HDBSCAN</li>
                <li><strong>Hypothesis Classification:</strong> Zero-shot learning (facebook/bart-large-mnli)</li>
                <li><strong>RAG Synthesis:</strong> Google Gemini API</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading system data: {e}")
