import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer



@st.cache_data
def load_data():
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
