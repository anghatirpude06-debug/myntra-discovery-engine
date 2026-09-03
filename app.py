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
Write a natural, flowing paragraph answer (2-4 sentences) that synthesizes the findings conversationally.
Do not use inline citation markers (like "Source 1", "Source 2") and do not list citations in the prose.

Question: {active_query}

User Feedback:
{context_str}
"""
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
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
    html, body, [class*="css"] { font-size: 15px !important; }
    h1 { font-size: 28px !important; font-weight: 700 !important; color: #29303E !important; }
    h2 { font-size: 24px !important; font-weight: 700 !important; color: #29303E !important; }
    h3 { font-size: 20px !important; font-weight: 600 !important; color: #29303E !important; }
    h4 { font-size: 16px !important; font-weight: 600 !important; color: #29303E !important; }
    h5, h6 { color: #29303E !important; }
    body, p, div { color: #5A6172; }
    .sub-label, small, .stCaption { color: #6B7280 !important; font-size: 13px !important; }
    
    .top-banner { background: #F7F8FA; border-left: 4px solid #F13AB1; padding: 28px 32px; border-radius: 0 0 24px 24px; box-shadow: 0 8px 16px rgba(0,0,0,0.15); margin-bottom: 24px; margin-top: 0; width: 100%; box-sizing: border-box; }
    
    .banner-card { background-color: #ffffff; border: 1px solid #E5E7EB; border-left: 4px solid #F13AB1; padding: 20px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .banner-card-pink { border-left-color: #E72744; }
    .banner-card h4 { color: #29303E !important; margin-top: 0; margin-bottom: 10px; }
    .banner-card p { margin-bottom: 0; }
    
    section[data-testid="stSidebar"] { background: #29303E !important; }
    section[data-testid="stSidebar"] button { height: 50px; border-radius: 12px; border: none !important; border-left: 2px solid transparent !important; background-color: transparent !important; color: #E8E9ED !important; font-weight: 600; justify-content: flex-start; padding-left: 16px; transition: all 0.2s ease; margin-bottom: 2px;}
    section[data-testid="stSidebar"] button p, section[data-testid="stSidebar"] button span, section[data-testid="stSidebar"] button div { color: inherit !important; transition: all 0.2s ease; }
    section[data-testid="stSidebar"] button:hover { background-color: #333B4C !important; color: #E8E9ED !important; border-left-color: #F05524 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    
    div.stButton > button { transition: all 0.2s ease; border-radius: 12px !important; }
    div.stButton > button:hover { box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-color: #F13AB1; color: #F13AB1; }
    
    /* Outlined secondary buttons */
    .stButton.quick-prompt > button { border: 1px solid #d1d5db !important; background-color: white !important; color: #29303E !important; }
    
    .opp-card { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB; border-left: 4px solid #F13AB1; box-shadow: 0 1px 3px rgba(0,0,0,0.06); height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .opp-card .opp-title { font-size: 13px; color: #5A6172; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .opp-card .opp-value { font-size: 18px; color: #E72744; font-weight: 700; line-height: 1.2; margin-bottom: 4px; }
    .opp-card .opp-sub { font-size: 13px; color: #5A6172; }
    
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB; border-left: 4px solid #F13AB1; box-shadow: 0 1px 3px rgba(0,0,0,0.06); transition: all 0.2s ease; }
    div[data-testid="stMetric"]:hover { box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-left-color: #FD913C; }
    div[data-testid="stMetricValue"] > div { color: #E72744 !important; }
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
        font-size: 24px !important; font-weight: 700 !important; color: #E8E9ED !important; text-align: left !important; margin-left: 0 !important;
    }
    
    /* Page background */
    .stApp { background-color: #f3f4f6 !important; }


    
    /* Chat bubbles */
    .chat-bubble-user { background-color: #29303E; color: #FFFFFF; padding: 12px 18px; border-radius: 12px; border-bottom-right-radius: 4px; display: inline-block; max-width: 80%; float: right; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); clear: both; }
    .chat-bubble-user p { color: #FFFFFF !important; margin: 0; }
    .chat-bubble-ai { background-color: #ffffff; border: 1px solid #E5E7EB; border-left: 4px solid #F13AB1; color: #5A6172; padding: 16px; border-radius: 12px; max-width: 90%; float: left; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); clear: both; }
    .chat-bubble-ai p { color: #5A6172 !important; margin-top: 0; }
    .chat-ai-label { font-size: 12px; color: #F13AB1; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    
    a.clear-chat-link:hover { color: #E72744 !important; }
    
    /* Hide markers completely from layout */
    div.element-container:has(> div > div > div[id$="-marker"]) {
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    
    .chat-container { display: flex; flex-direction: column; width: 100%; margin-bottom: 20px; }
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
        with open("streamlit_app/data/precomputed_answers.json", "r", encoding="utf-8") as f:
            precomputed_answers = json.load(f)
    except Exception:
        precomputed_answers = {}
    
    if 'active_page' not in st.session_state:
        st.session_state['active_page'] = "Ask the Engine"
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    with st.sidebar:
        if st.button("About"):
            st.session_state['active_page'] = "Ask the Engine"
            st.rerun()
        st.markdown("<p class='sub-label' style='font-size: 13px; margin-top: -10px;'>This intelligence engine parses unstructured social media and app store feedback to expose the hidden friction points causing wishlist abandonment.</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### Navigation")
        
        pages = ["Ask the Engine", "Deep Analytics", "Methodology"]
        
        for i, page in enumerate(pages):
            if st.button(f"{page}", key=f"nav_{page}", use_container_width=True):
                st.session_state['active_page'] = page
                st.rerun()
                


        # Push Sync Data to the bottom of the sidebar visually
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
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

    import streamlit.components.v1 as components
    selection = st.session_state['active_page']
    
    st.markdown("""
    <div class="top-banner">
        <h1 style="margin-top: 0; margin-bottom: 12px; color: #29303E !important; font-weight: 800; font-size: 32px !important;">
            MyntraLens
        </h1>
        <p style="margin-top: 0; font-size: 16px !important; font-weight: 400; line-height: 1.6; color: #5A6172; margin-bottom: 0;">
            MyntraLens analyzes <b style="color: #29303E;">6,181 real reviews and posts</b> (5,014 Play Store · 702 App Store · 465 Reddit) to uncover why Myntra users don't convert wishlisted items into purchases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if selection == "Ask the Engine":
        if "clear" in st.query_params:
            st.session_state['chat_history'] = []
            st.query_params.clear()
            st.rerun()
            
        st.header("Ask the Engine")
        st.markdown("<p class='sub-label'>Use our RAG-powered engine to search the raw review dataset and generate synthesized answers based on actual customer complaints.</p>", unsafe_allow_html=True)

        
        # New styling explicitly scoped to Generate Insight button
        st.markdown("""
        <style>
            div[data-testid="stFormSubmitButton"] > button { background-color: #29303E !important; color: #FFFFFF !important; font-weight: bold !important; border: none !important; border-radius: 12px !important; }
            div[data-testid="stFormSubmitButton"] > button:hover { background-color: #333B4C !important; color: #FFFFFF !important; border: none !important; }
            div[data-testid="stFormSubmitButton"] > button p, div[data-testid="stFormSubmitButton"] > button div, div[data-testid="stFormSubmitButton"] > button span { color: #FFFFFF !important; font-weight: bold !important; }
        </style>
        """, unsafe_allow_html=True)
        
        q1 = "Why do users hesitate before buying wishlisted items?"
        q2 = "What causes the most return and refund complaints?"
        q3 = "Do users trust Myntra's product authenticity?"
        q4 = "What delivery issues do users report most often?"
        q5 = "What UX problems affect the wishlist feature?"
        
        active_query = None
        if 'submit_query' in st.session_state and st.session_state['submit_query']:
            active_query = st.session_state['submit_query']
            st.session_state['submit_query'] = None
            
        def render_input_row():
            with st.form("chat_form", clear_on_submit=True):
                fc1, fc2 = st.columns([4, 1])
                with fc1:
                    custom_query = st.text_input("Query", label_visibility="collapsed", placeholder="Ask about wishlist behavior, returns, sizing...")
                with fc2:
                    submitted = st.form_submit_button("Generate Insight", type="primary", use_container_width=True)
            st.markdown("<p style='font-size: 12px; color: #9CA3AF; font-style: italic; margin-top: -10px; margin-bottom: 5px;'>AI-generated responses may contain inaccuracies.</p>", unsafe_allow_html=True)
            return submitted, custom_query

        if len(st.session_state['chat_history']) == 0:
            # STATE 1: Empty History
            st.markdown("<h3 style='text-align: center; margin-top: 40px;'>Suggested Questions</h3>", unsafe_allow_html=True)
            
            main_suggestions_card = st.container()
            with main_suggestions_card:
                st.markdown("<div id='main-suggestions-marker'></div>", unsafe_allow_html=True)
                st.markdown("""
                <style>
                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) {
                        flex-direction: row !important;
                        flex-wrap: wrap !important;
                        justify-content: center !important;
                        gap: 12px !important;
                    }
                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) div.element-container,
                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) div[data-testid="stButton"] {
                        width: auto !important;
                        flex: 0 0 auto !important;
                    }
                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) div[data-testid="stButton"] button {
                        font-size: 13.5px !important; 
                        margin-bottom: 0px !important; 
                        white-space: nowrap !important; 
                        height: auto !important; 
                        padding: 10px 16px !important; 
                        border-radius: 50px !important; 
                        background-color: #ffffff !important; 
                        border: 1px solid #e5e7eb !important; 
                        color: #5A6172 !important; 
                        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
                        width: auto !important;
                        display: inline-flex !important;
                        align-items: center !important;
                    }
                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) div[data-testid="stButton"] button * {
                        margin: 0 !important;
                        padding: 0 !important;
                        width: auto !important;
                    }

                    div[data-testid="stVerticalBlock"]:has(> div.element-container #main-suggestions-marker) div[data-testid="stButton"] button:hover { 
                        border-color: #F13AB1 !important; 
                        color: #F13AB1 !important; 
                    }
                </style>
                """, unsafe_allow_html=True)
                if st.button(q1, key="q1_main"): active_query = q1
                if st.button(q2, key="q2_main"): active_query = q2
                if st.button(q3, key="q3_main"): active_query = q3
                if st.button(q4, key="q4_main"): active_query = q4
                if st.button(q5, key="q5_main"): active_query = q5
                
            st.write("")
            st.write("")
            submitted, custom_query = render_input_row()
            if submitted and custom_query:
                active_query = custom_query
        else:
            # STATE 2: Active Chat Log
            col_main, col_suggestions = st.columns([3, 1])
            
            with col_main:
                main_chat_card = st.container(border=True)
                with main_chat_card:
                    st.markdown("""
                    <div id='main-chat-card-marker'></div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-bottom: 12px;">
                        <h5 class='chat-log-title' style='margin: 0; padding: 0; line-height: 1.5; color: #29303E; font-weight: 600;'>CHAT LOG</h5>
                        <a href="?clear=true" target="_self" class="clear-chat-link" style="color: #5A6172; font-size: 14px; font-weight: 600; text-decoration: underline; margin: 0; line-height: 1.5;">Clear Chat</a>
                    </div>
                    <hr style='margin: 0 0 10px 0; border: none; border-top: 1px solid #e5e7eb;'/>
                    """, unsafe_allow_html=True)
                    
                    chat_container = st.container(height=550, border=False)
                    with chat_container:
                        for msg in st.session_state['chat_history']:
                            if msg['role'] == 'user':
                                st.markdown(f"""
                                <div class="chat-container">
                                    <div class="chat-bubble-user">
                                        <p>{msg['content']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="chat-container">
                                    <div class="chat-bubble-ai">
                                        <div class="chat-ai-label">🤖 AI ASSISTANT RESPONSE</div>
                                        <p>{msg['content']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                if 'sources' in msg and msg['sources']:
                                    with st.expander("View supporting reviews"):
                                        for src in msg['sources']:
                                            st.markdown(f"- {src}")
                        
                        wait_placeholder = st.empty()
                        
                        st.markdown("<div id='chat-end-marker'></div>", unsafe_allow_html=True)
                        
                        if 'last_chat_len' not in st.session_state or st.session_state['last_chat_len'] != len(st.session_state['chat_history']):
                            st.session_state['last_chat_len'] = len(st.session_state['chat_history'])
                            components.html("""
                                <script>
                                    const marker = window.parent.document.getElementById('chat-end-marker');
                                    if (marker) {
                                        marker.scrollIntoView({behavior: 'smooth', block: 'end'});
                                    }
                                </script>
                            """, height=0)
                
                input_card = st.container()
                with input_card:
                    st.markdown("<div id='input-card-marker'></div>", unsafe_allow_html=True)
                    submitted, custom_query = render_input_row()
                    if submitted and custom_query:
                        active_query = custom_query
                        
            with col_suggestions:
                sidebar_card = st.container()
                with sidebar_card:
                    st.markdown("""
                    <div id='sidebar-card-marker'></div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-bottom: 12px;">
                        <h5 style='margin: 0; padding: 0; line-height: 1.5; color: #29303E !important; font-weight: 700; text-transform: uppercase;'>SUGGESTED QUESTIONS</h5>
                    </div>
                    <hr style='margin: 0 0 10px 0; border: none; border-top: 1px solid #e5e7eb;'/>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <style>
                        div[data-testid="stVerticalBlock"]:has(> div.element-container #sidebar-card-marker) {
                            background-color: #FFFFFF !important;
                            border-radius: 12px !important;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
                            border: 1px solid #e5e7eb !important;
                            padding: 24px !important;
                        }
                        div[data-testid="stVerticalBlock"]:has(#sidebar-card-marker) div[data-testid="stButton"] button {
                            text-align: left !important; 
                            justify-content: flex-start !important; 
                            font-size: 13px !important; 
                            margin-bottom: 8px !important; 
                            white-space: normal !important; 
                            height: auto !important; 
                            padding: 12px 16px !important; 
                            line-height: 1.4 !important; 
                            border-radius: 12px !important; 
                            background-color: #ffffff !important; 
                            border: 1px solid #e5e7eb !important; 
                            color: #5A6172 !important; 
                            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
                            width: 100% !important;
                            display: flex !important;
                        }
                        div[data-testid="stVerticalBlock"]:has(#sidebar-card-marker) div[data-testid="stButton"] button * {
                            text-align: left !important;
                            justify-content: flex-start !important;
                            width: 100% !important;
                            margin: 0 !important;
                        }

                        div[data-testid="stVerticalBlock"]:has(#sidebar-card-marker) div[data-testid="stButton"] button:hover { 
                            border-color: #F13AB1 !important; 
                            color: #F13AB1 !important; 
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    if st.button(q1, use_container_width=True, key="q1_side"): active_query = q1
                    if st.button(q2, use_container_width=True, key="q2_side"): active_query = q2
                    if st.button(q3, use_container_width=True, key="q3_side"): active_query = q3
                    if st.button(q4, use_container_width=True, key="q4_side"): active_query = q4
                    if st.button(q5, use_container_width=True, key="q5_side"): active_query = q5
                
        # Handle Query Processing
        if active_query:
            if not active_query.strip():
                st.warning("Please type a valid question.")
            else:
                st.session_state['chat_history'].append({"role": "user", "content": active_query})
                
                # Check for cached/precomputed answers first to speed up known questions
                if active_query in precomputed_answers:
                    ans_data = precomputed_answers[active_query]
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": ans_data['answer'],
                        "sources": ans_data['sources'],
                        "is_new": True
                    })
                    st.rerun()
                else:
                    # Not in precomputed, do full RAG
                    with st.spinner("Generating insight..."):
                        try:
                            embedder = load_embedder()
                            texts = clustered['text'].astype(str).tolist()
                            corpus_embeddings = get_corpus_embeddings(embedder, texts)
                            
                            query_embedding = embedder.encode(active_query, convert_to_tensor=True)
                            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=8)[0]
                            
                            if not hits or hits[0]['score'] < 0.4:
                                st.session_state['chat_history'].append({
                                    "role": "assistant",
                                    "content": "No relevant insights found for that question. Try asking about fit, sizing, returns, delivery, trust, pricing, or the wishlist feature.",
                                    "sources": [],
                                    "is_new": True
                                })
                                st.rerun()
                            else:
                                context_texts = []
                                for i, hit in enumerate(hits):
                                    idx = hit['corpus_id']
                                    row = clustered.iloc[idx]
                                    context_texts.append(f"({row.get('source', 'N/A')}): {row.get('text', 'N/A')}")
                                    
                                context_str = "\\n\\n".join(context_texts)
                                
                                try:
                                    text_out, _ = generate_rag_insight(active_query, context_str)
                                    
                                    if text_out == "QUOTA_ERROR":
                                        st.session_state['chat_history'].append({
                                            "role": "assistant",
                                            "content": "⚠️ **API Quota Exceeded**: The Gemini API rate limit (per-minute or daily) has been reached. Please wait a moment and try again, or use one of the precomputed suggested questions.",
                                            "sources": [],
                                            "is_new": True
                                        })
                                    else:
                                        st.session_state['chat_history'].append({
                                            "role": "assistant",
                                            "content": text_out,
                                            "sources": context_texts,
                                            "is_new": True
                                        })
                                    st.rerun()
                                except Exception as e:
                                    st.session_state['chat_history'].append({
                                        "role": "assistant",
                                        "content": f"⚠️ **An error occurred during retrieval**: {str(e)}",
                                        "sources": [],
                                        "is_new": True
                                    })
                                    st.rerun()
                        except Exception:
                            st.session_state['chat_history'].append({
                                "role": "assistant",
                                "content": "No relevant insights found for that question. Try asking about fit, sizing, returns, delivery, trust, pricing, or the wishlist feature.",
                                "sources": [],
                                "is_new": True
                            })
                            st.rerun()
                            
        # 15s delay logic has been entirely removed

    elif selection == "Deep Analytics":
        st.header("Deep Analytics")
        st.markdown("<p class='sub-label'>A breakdown of the primary reasons users hesitate and abandon their wishlists.</p>", unsafe_allow_html=True)
        
        tab_themes, tab_insights = st.tabs(["Themes", "Validated Insights"])
        
        with tab_insights:
            st.markdown("""
            <div class="banner-card banner-card-pink">
                <h4 style="color: #F13AB1 !important;">The Validated Finding</h4>
                <p style="font-size: 16px;"><strong>Users don't abandon wishlists because they lose interest—they abandon them because they lack trust in the post-purchase experience.</strong><br>Across the dataset, users are fundamentally struggling with post-purchase anxiety that bleeds into their pre-purchase wishlist behavior. The highest volume of friction stems directly from complicated return processes, delayed refunds, and lingering doubts regarding product authenticity. When users don't trust the fulfillment or return pipeline, they use the wishlist as a holding area rather than converting to a cart.</p>
            </div>
            """, unsafe_allow_html=True)
            

            st.subheader("The gap, in users' words")
            st.markdown("""
            > *"I have 50 items in my wishlist but I'm scared to order because the last time I returned something it took a month to get my money back."*
            
            > *"Love the clothes but the sizes are so inconsistent. I just wishlist them and wait to see if I can find them in store instead of dealing with returns."*
            
            > *"Customer service is unresponsive when items arrive damaged. Keeping items in wishlist forever because I don't want to risk another bad experience."*
            """)
            

            st.subheader("Cross-source triangulation")
            if not ranking.empty and 'source' in tagged.columns:
                source_dist = pd.crosstab(tagged['hypothesis_tag'], tagged['source'], normalize='index') * 100
                source_dist = source_dist.round(1)
                
                display_dist = source_dist.copy()
                display_dist.index = display_dist.index.map(lambda x: TAG_MAP.get(x, x))
                
                for col in display_dist.columns:
                    display_dist[col] = display_dist[col].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_dist, use_container_width=True)
            else:
                st.info("Source distribution data not available.")
                

            st.subheader("Why this matters for the growth goal")
            st.markdown("""
            * **Returns & Refunds are the real conversion killer:** Fixing the cart experience won't help if users are deterred by return anxiety before they even add to cart.
            * **Trust signals are missing:** Users need authenticity guarantees and transparent return policies visible *within* the wishlist itself.
            * **Sizing uncertainty creates friction:** Predictive sizing would give users the confidence to move items from wishlist to cart.
            * **This is a hypothesis to validate in interviews, not a conclusion — reviews can't observe wishlist abandonment directly.**
            """)
            

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
                
    
                st.markdown(f"""
                <div class="survey-metric" style="flex-direction: row; justify-content: space-between; align-items: center; text-align: left; margin-top: 15px; width: 100%;">
                    <div>
                        <div class="lbl" style="text-transform: uppercase; margin-bottom: 5px;">Top reason unpurchased</div>
                        <div class="text-val" style="font-size: 20px; color: #282C3F; margin-bottom: 0;">{top_reason}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                    
            else:
                st.markdown("""
                <div class="banner-card">
                    <h4 style="color: #6B7280 !important;">Survey data pending</h4>
                    <p>This section will be integrated directly from primary qualitative research.</p>
                </div>
                """, unsafe_allow_html=True)
            
        with tab_themes:
            if not ranking.empty:
                valid_ranking = ranking[ranking['hypothesis_tag'] != "general positive satisfaction with no specific complaint"].copy()
                valid_ranking['display_name'] = valid_ranking['hypothesis_tag'].map(TAG_MAP).fillna(valid_ranking['hypothesis_tag'].astype(str))
                
                fig_ranking = px.bar(valid_ranking, x='percentage_of_total', y='display_name', orientation='h',
                                     color='avg_negativity_intensity', color_continuous_scale=[[0, "#FF7A1A"], [1, "#F5088B"]],
                                     text='percentage_of_total', title="What users actually talk about")
                fig_ranking.update_traces(hovertemplate='<b>%{y}</b><br>Percentage: %{x:.1f}%', texttemplate='%{text:.1f}%', textposition='outside')
                fig_ranking.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="% of Reviews", yaxis_title="")
                st.plotly_chart(fig_ranking, use_container_width=True)
                
                st.markdown("*Read this chart carefully. Returns and fulfillment dominate. Wishlist-related themes are tiny. Users rarely explicitly state why they abandoned a wishlist item in a review; instead, they complain about a bad experience (like complicated returns), which we infer is the root cause for their hesitation on future purchases.*")
                
    
                st.subheader("Drill into a theme")
                selected_theme = st.selectbox("Select a theme to explore", valid_ranking['display_name'].tolist())
                
                if selected_theme:
                    original_tag = valid_ranking[valid_ranking['display_name'] == selected_theme]['hypothesis_tag'].iloc[0]
                    theme_data = tagged[tagged['hypothesis_tag'] == original_tag]
                    theme_pct = valid_ranking[valid_ranking['display_name'] == selected_theme]['percentage_of_total'].iloc[0]
                    
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.metric("Share of reviews", f"{theme_pct:.1f}%")
                    with t2:
                        st.metric("Play Store", len(theme_data[theme_data['source'] == 'play_store']))
                    with t3:
                        st.metric("App Store", len(theme_data[theme_data['source'] == 'app_store']))
                    with t4:
                        st.metric("Reddit", len(theme_data[theme_data['source'] == 'reddit']))
                        
        
                    st.markdown(f"**Representative voices for '{selected_theme}'**")
                    sample_reviews = theme_data['text'].dropna().sample(min(5, len(theme_data)), random_state=42).tolist()
                    for txt in sample_reviews:
                        st.markdown(f"> \"{txt}\"")

    elif selection == "Methodology":
        st.header("Methodology")
        
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
        
    st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 12px; margin-top: 60px;'>MyntraLens · built for a NextLeap Growth PM case study · data is user-generated app-store & social feedback.</p>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading system data: {e}")
