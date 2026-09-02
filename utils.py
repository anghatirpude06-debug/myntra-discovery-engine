import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

def inject_custom_css():
    st.markdown("""
    <style>
        /* Primary Magenta: #F5088B, Accent Orange: #FF7A1A, Body: #282C3F */
        h1, h2, h3 {
            color: #F5088B !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: #F5088B !important;
        }
        /* Light styling to cards */
        div[data-testid="stMetric"] {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #F5088B;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }
        .stButton>button {
            color: #282C3F;
            border: 1px solid #F5088B;
        }
        .stButton>button:hover {
            background-color: #FF7A1A;
            border: 1px solid #FF7A1A;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        tagged = pd.read_csv("streamlit_app/data/tagged_dataset.csv")
        clustered = pd.read_csv("streamlit_app/data/clustered_dataset.csv")
        ranking = pd.read_csv("streamlit_app/data/opportunity_ranking_filtered.csv")
        return tagged, clustered, ranking
    except FileNotFoundError:
        tagged = pd.read_csv("data/tagged_dataset.csv")
        clustered = pd.read_csv("data/clustered_dataset.csv")
        ranking = pd.read_csv("data/opportunity_ranking_filtered.csv")
        return tagged, clustered, ranking

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_corpus_embeddings(_model, _texts):
    return _model.encode(_texts, convert_to_tensor=True)
