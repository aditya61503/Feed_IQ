import streamlit as st
import pandas as pd
from ml_engine import MLEngine
from data_manager_fixed import DataManager

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title='FeedIQ - AI Feedback Intelligence',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================
# CUSTOM CSS FOR NEUTRAL DESIGN
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }
    
    /* Custom header styling */
    .main-header {
        background-color: #ffffff;
        padding: 2rem;
        border-bottom: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #111827;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Button styling (Neutral) */
    .stButton > button {
        background-color: #f3f4f6;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
        color: #111827;
    }
    
    /* Primary Action Buttons */
    .primary-btn {
        background-color: #2563eb !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #6b7280;
        font-weight: 500;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }
    
    /* Priority badges */
    .priority-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    
    .priority-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    
    .priority-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #111827;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6b7280;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 8px;
    }
    
    /* Section headers */
    .section-header {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE
# ============================================
manager = DataManager()
engine = MLEngine()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>FeedIQ</h1>
    <p>Feedback Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="Upload a CSV with a 'text' or 'feedback' column")
    
    st.markdown("---")

    # Only show submission form if using default dataset
    if uploaded_file is None:
        st.markdown("### Submit Feedback")
        
        feedback = st.text_area(
            'Enter feedback',
            placeholder='Type here...',
            height=150,
            label_visibility='collapsed'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Submit', use_container_width=True):
                if feedback:
                    manager.add_feedback(feedback)
                    st.success('Added!')
                    st.rerun()
                else:
                    st.warning('Empty feedback')
        
        with col2:
            if st.button('Refresh', use_container_width=True):
                st.rerun()
        
        st.markdown("---")

    st.markdown("### Stats")

# ============================================
# LOAD AND PROCESS DATA
# ============================================
if uploaded_file is not None:
    try:
        # Initial read
        df = pd.read_csv(uploaded_file)
        
        # Check if first row looks like data (heuristic: partial date match or many unique values)
        # If the first column name looks like a sentence, it's likely a missing header
        first_col = df.columns[0]
        if ' ' in first_col and len(first_col) > 20:
             # Reload with header=None
             uploaded_file.seek(0)
             df = pd.read_csv(uploaded_file, header=None)
             st.sidebar.info("Detected headerless CSV. Reloaded.")
        
        # Smart column detection - prioritize columns with actual text
        text_col = None
        candidates = ['text', 'feedback', 'review', 'comment', 'content', 'body', 'desc']
        
        # 1. Search for candidate names
        for col in df.columns:
            if str(col).lower() in candidates:
                text_col = col
                break
        
        # 2. If no candidate, look for the column with the longest unique strings (heuristic for feedback)
        if text_col is None:
            max_avg_len = 0
            for col in df.columns:
                # Skip numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Check average string length
                try:
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > max_avg_len and avg_len > 10:  # Threshold for "sentence-like" data
                        max_avg_len = avg_len
                        text_col = col
                except:
                    continue

        # If still no match, use first non-numeric column or first column
        if text_col is None:
            non_numeric = df.select_dtypes(exclude=['number']).columns
            text_col = non_numeric[0] if len(non_numeric) > 0 else df.columns[0]
            st.sidebar.warning(f"Column not found. Using '{text_col}' as feedback.")
        
        # Standardize column name
        df = df.rename(columns={text_col: 'text'})
        
        # CLEANING: Ensure text column is string, handle NaNs, and remove numbers/special chars
        df['text'] = df['text'].fillna('').astype(str)
        # Regex to keep only letters and spaces (User request: "just consider characters")
        df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

        
        # Ensure ID column exists
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            
        st.sidebar.success(f"Loaded {len(df)} rows")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    # Load default data
    df = manager.load()

# Process Data
texts = df['text'].astype(str).values
X = engine.vectorize(texts)
clusters = engine.cluster(X)
sim_matrix = engine.similarity(X)

df['cluster'] = clusters
cluster_names = engine.name_clusters(X, texts, clusters)
df['cluster_name'] = df['cluster'].map(cluster_names)

scores = [engine.priority_score(sim_matrix, i) for i in range(len(texts))]
df['priority_score'] = scores
df['priority_level'] = df['priority_score'].apply(engine.priority_level)
df['tags'] = df['text'].apply(engine.generate_tags)

# ============================================
# SIDEBAR STATS
# ============================================
with st.sidebar:
    st.metric("Total", len(df))
    st.metric("High Priority", len(df[df['priority_level'] == 'High']))
    st.metric("Categories", len(df['cluster_name'].unique()))

# ============================================
# MAIN DASHBOARD
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Feedbacks", len(df))

with col2:
    high_priority = len(df[df['priority_level'] == 'High'])
    st.metric("High Priority", high_priority)

with col3:
    st.metric("Categories", len(df['cluster_name'].unique()))

with col4:
    avg_score = df['priority_score'].mean()
    st.metric("Avg Score", f"{avg_score:.2f}")

st.markdown("---")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "All Feedback",
    "Priority Issues",
    "Categories",
    "Similar Finder"
])

# ============================================
# TAB 1: ALL FEEDBACK
# ============================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        filter_category = st.multiselect(
            "Filter Category",
            options=df['cluster_name'].unique(),
            default=df['cluster_name'].unique()
        )
    with col2:
        filter_priority = st.multiselect(
            "Filter Priority",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    
    filtered_df = df[
        (df['cluster_name'].isin(filter_category)) &
        (df['priority_level'].isin(filter_priority))
    ]
    
    st.markdown(f"**{len(filtered_df)} feedbacks**")
    
    display_df = filtered_df[['text', 'cluster_name', 'priority_level', 'tags', 'priority_score']].copy()
    display_df.columns = ['Feedback', 'Category', 'Priority', 'Tags', 'Score']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )

# ============================================
# TAB 2: TOP PRIORITY ISSUES
# ============================================
with tab2:
    st.markdown("### Top Priority Items")
    
    num_top = st.slider("Count", 5, 20, 10)
    top_df = df.sort_values(by='priority_score', ascending=False).head(num_top)
    
    for idx, row in top_df.iterrows():
        priority_class = f"priority-{row['priority_level'].lower()}"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['text']}**")
                st.caption(f"{row['cluster_name']} • {', '.join(row['tags'][:3])}")
            with col2:
                st.markdown(f'<span class="{priority_class}">{row["priority_level"]}</span>', unsafe_allow_html=True)
            st.divider()

# ============================================
# TAB 3: CATEGORIES
# ============================================
with tab3:
    st.markdown("### Category Grouping")
    
    summary = engine.generate_summary(df)
    st.info(summary)
    
    for name in sorted(df['cluster_name'].unique()):
        category_df = df[df['cluster_name'] == name]
        
        with st.expander(f"{name} ({len(category_df)})"):
            st.dataframe(
                category_df[['text', 'priority_level']],
                use_container_width=True,
                hide_index=True
            )

# ============================================
# TAB 4: SIMILAR FEEDBACK
# ============================================
with tab4:
    st.markdown("### Find Similar")
    
    selected = st.selectbox(
        "Select feedback",
        df['text'].values,
    )
    
    if selected:
        idx = list(df['text'].values).index(selected)
        similar = engine.find_similar(sim_matrix, texts, idx)
        
        st.markdown("**Selected:**")
        st.info(selected)
        
        st.markdown("**Similar Items:**")
        for i, s in enumerate(similar, 1):
            st.markdown(f"{i}. {s}")
