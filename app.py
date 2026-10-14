"""
Product Rating Analysis using Data Analytics
A modern, modular Streamlit web application for e-commerce catalog and customer review analysis.
Suitable for College Minor Projects, Capstone Demonstrations, and Viva Presentations.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import streamlit as st

# Local modules
from src.data_loader import load_and_clean_data
from src.analytics import (
    compute_kpi_metrics,
    compute_category_analytics,
    get_top_products,
    get_lowest_products,
    compute_correlations,
    generate_data_driven_insights,
)
from src.sentiment import SentimentAnalyzer, extract_key_terms, get_sentiment_limitations
from src.visualizer import (
    plot_rating_distribution,
    plot_category_ratings,
    plot_category_reviews,
    plot_price_vs_rating,
    plot_top_and_bottom_products,
    plot_sentiment_breakdown,
)


# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="Product Rating Analysis | Data Analytics",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Container Styles */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #F8FAFC;
        margin: 0 0 6px 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94A3B8;
        margin: 0;
        font-size: 14px;
    }
    .badge-tag {
        display: inline-block;
        background-color: #3B82F6;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 26px;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #10B981;
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
    }
    .metric-sub-warning {
        color: #EF4444;
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Info Callouts */
    .insight-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .insight-card-warning {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .insight-card-success {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- 2. Data Ingestion & Caching ---
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_products.csv")

@st.cache_data(show_spinner=False)
def get_dataset(file_source: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and scrubs raw CSV inputs with caching for performance."""
    return load_and_clean_data(file_source)


@st.cache_resource(show_spinner=False)
def get_sentiment_engine() -> SentimentAnalyzer:
    """Initializes and caches the sentiment analysis engine."""
    return SentimentAnalyzer()


# --- 3. Sidebar Controls & Dynamic Filters ---
with st.sidebar:
    st.markdown("### 🛒 Product Rating Analytics")
    st.caption("College Minor Project | Data Analytics")
    st.markdown("---")

    # Ingestion Source Selector
    st.markdown("#### 📂 1. Data Source")
    data_source_option = st.radio(
        "Choose Data Source:",
        ["Built-in Sample Dataset", "Upload Custom CSV"],
        index=0,
        help="Use preloaded catalog data or upload your own CSV file."
    )

    source_input = DEFAULT_DATA_PATH
    if data_source_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Product Reviews CSV",
            type=["csv"],
            help="Expected columns: product_name, category, rating, num_reviews, price, review_text"
        )
        if uploaded_file is not None:
            source_input = uploaded_file
        else:
            st.info("Awaiting CSV upload. Falling back to default sample dataset.")
            source_input = DEFAULT_DATA_PATH

    # Load cleaned data
    try:
        raw_clean_df, clean_stats = get_dataset(source_input)
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        st.stop()

    if raw_clean_df.empty:
        st.warning("The selected dataset is empty or invalid.")
        st.stop()

    st.markdown("---")
    st.markdown("#### 🔍 2. Filter Products")

    # Category Filter
    all_categories = sorted(raw_clean_df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Select Categories:",
        options=all_categories,
        default=all_categories,
        help="Filter products across one or more departments."
    )

    # Rating Range Slider
    min_rating_val = float(raw_clean_df["rating"].min())
    max_rating_val = float(raw_clean_df["rating"].max())
    rating_range = st.slider(
        "Rating Range (Stars):",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.1,
        help="Filter products by star rating bounds."
    )

    # Price Range Slider
    min_price_val = float(raw_clean_df["price"].min())
    max_price_val = float(raw_clean_df["price"].max())
    price_range = st.slider(
        "Price Range ($):",
        min_value=round(min_price_val, 2),
        max_value=round(max_price_val, 2),
        value=(round(min_price_val, 2), round(max_price_val, 2)),
        step=1.0,
        help="Filter products by price range."
    )

    # Minimum Review Count Filter
    max_reviews_val = int(raw_clean_df["num_reviews"].max())
    min_reviews_threshold = st.slider(
        "Minimum Review Count:",
        min_value=0,
        max_value=min(max_reviews_val, 2000),
        value=0,
        step=50,
        help="Filter out low-volume products to focus on established items."
    )

    st.markdown("---")
    # Cleaning Statistics Callout
    with st.expander("🛡️ Data Cleaning & Hygiene Report", expanded=False):
        st.markdown(f"- **Raw Ingested Records:** `{clean_stats['raw_count']}`")
        st.markdown(f"- **Duplicates Removed:** `{clean_stats['duplicates_removed']}`")
        st.markdown(f"- **Missing Value Rows Dropped:** `{clean_stats['missing_critical_dropped']}`")
        st.markdown(f"- **Out-of-Bounds Ratings Filtered:** `{clean_stats['out_of_bounds_ratings']}`")
        st.markdown(f"- **Final Clean Records:** `{clean_stats['final_clean_count']}`")


# --- 4. Apply Filters to Create Working View ---
active_df = raw_clean_df[
    (raw_clean_df["category"].isin(selected_categories)) &
    (raw_clean_df["rating"] >= rating_range[0]) &
    (raw_clean_df["rating"] <= rating_range[1]) &
    (raw_clean_df["price"] >= price_range[0]) &
    (raw_clean_df["price"] <= price_range[1]) &
    (raw_clean_df["num_reviews"] >= min_reviews_threshold)
].copy().reset_index(drop=True)


# --- 5. Main Dashboard Header ---
st.markdown(
    """
    <div class="main-header">
        <span class="badge-tag">Academic Minor Project</span>
        <h1>Product Rating Analysis using Data Analytics</h1>
        <p>Interactive exploratory dashboard, statistical evaluation, sentiment mining, and business intelligence reporting.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if active_df.empty:
    st.warning("⚠️ No products match the selected filters. Please broaden your filter criteria in the sidebar.")
    st.stop()


# --- 6. Executive KPI Metrics Cards ---
kpi_data = compute_kpi_metrics(active_df, low_threshold=3.0)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Products</div>
            <div class="metric-value">{kpi_data['total_products']}</div>
            <div class="metric-sub">Catalog Sample</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Rating</div>
            <div class="metric-value">{kpi_data['avg_rating']:.2f} <span style="font-size:16px;color:#F59E0B;">★</span></div>
            <div class="metric-sub">Std Dev: ±{kpi_data['rating_std']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Reviews</div>
            <div class="metric-value">{kpi_data['total_reviews']:,}</div>
            <div class="metric-sub">Avg {kpi_data['avg_reviews']} / item</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    low_pct = kpi_data['low_rated_percentage']
    sub_class = "metric-sub-warning" if low_pct > 15 else "metric-sub"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Low-Rated Share</div>
            <div class="metric-value">{low_pct}%</div>
            <div class="{sub_class}">{kpi_data['low_rated_count']} items (&lt;3.0 ★)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value">${kpi_data['avg_price']:.2f}</div>
            <div class="metric-sub">Unit Mean Cost</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# --- 7. Tabbed Interactive Sections ---
tab_overview, tab_visuals, tab_rankings, tab_sentiment, tab_explorer, tab_viva = st.tabs([
    "📊 Executive Overview",
    "📈 Visual Analytics",
    "🏆 Product Rankings",
    "💬 Review Sentiment",
    "🔍 Data Explorer",
    "🎓 Project & Viva Guide"
])


# --- TAB 1: EXECUTIVE OVERVIEW & INSIGHTS ---
with tab_overview:
    st.subheader("💡 Data-Driven Analytical Insights")
    st.caption("Automated findings calculated dynamically from active catalog filters.")

    insights = generate_data_driven_insights(active_df, low_threshold=3.0)
    for item in insights:
        css_type = "insight-card"
        if item.get("type") == "warning":
            css_type = "insight-card-warning"
        elif item.get("type") == "success":
            css_type = "insight-card-success"

        st.markdown(
            f"""
            <div class="{css_type}">
                <strong style="color: #0F172A; font-size: 14.5px;">{item['title']}</strong><br/>
                <span style="color: #334155; font-size: 13.5px;">{item['text']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📋 Category Performance Summary Table")
    cat_summary_df = compute_category_analytics(active_df)

    # Clean display columns
    display_cat = cat_summary_df.rename(columns={
        "category": "Category",
        "product_count": "Products",
        "avg_rating": "Mean Rating",
        "median_rating": "Median Rating",
        "total_reviews": "Total Reviews",
        "avg_reviews": "Avg Reviews / Item",
        "avg_price": "Mean Price ($)",
        "low_rated_pct": "Low-Rated Share (%)"
    })
    st.dataframe(
        display_cat.style.format({
            "Mean Rating": "{:.2f}",
            "Median Rating": "{:.2f}",
            "Total Reviews": "{:,.0f}",
            "Avg Reviews / Item": "{:.1f}",
            "Mean Price ($)": "${:.2f}",
            "Low-Rated Share (%)": "{:.1f}%"
        }),
        use_container_width=True
    )


# --- TAB 2: VISUAL ANALYTICS ---
with tab_visuals:
    st.subheader("📈 Statistical Distributions & Category Dynamics")
    st.caption("Visualizing rating spreads, review volume, and price correlation.")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig_dist = plot_rating_distribution(active_df)
        st.pyplot(fig_dist, use_container_width=True)
    with row1_col2:
        fig_cat_rate = plot_category_ratings(cat_summary_df)
        st.pyplot(fig_cat_rate, use_container_width=True)

    st.markdown("---")
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        fig_cat_rev = plot_category_reviews(cat_summary_df)
        st.pyplot(fig_cat_rev, use_container_width=True)
    with row2_col2:
        fig_price_rate = plot_price_vs_rating(active_df)
        st.pyplot(fig_price_rate, use_container_width=True)


# --- TAB 3: PRODUCT RANKINGS & LEADERBOARD ---
with tab_rankings:
    st.subheader("🏆 Product Performance Rankings")
    st.caption("Identify benchmark best-sellers and inventory requiring quality remediation.")

    top_products_df = get_top_products(active_df, top_n=10, min_reviews=min_reviews_threshold)
    bottom_products_df = get_lowest_products(active_df, bottom_n=10, min_reviews=min_reviews_threshold)

    # Subplots visualization
    fig_rankings = plot_top_and_bottom_products(top_products_df, bottom_products_df)
    st.pyplot(fig_rankings, use_container_width=True)

    st.markdown("---")
    rank_col1, rank_col2 = st.columns(2)

    with rank_col1:
        st.markdown("#### ⭐ Top 10 Best Rated Products")
        if not top_products_df.empty:
            st.dataframe(
                top_products_df.rename(columns={
                    "product_name": "Product Name",
                    "category": "Category",
                    "rating": "Rating",
                    "num_reviews": "Reviews",
                    "price": "Price ($)"
                }).style.format({
                    "Rating": "{:.2f} ★",
                    "Reviews": "{:,}",
                    "Price ($)": "${:.2f}"
                }),
                use_container_width=True
            )
        else:
            st.info("No products match criteria.")

    with rank_col2:
        st.markdown("#### ⚠️ Bottom 10 Underperforming Products")
        if not bottom_products_df.empty:
            st.dataframe(
                bottom_products_df.rename(columns={
                    "product_name": "Product Name",
                    "category": "Category",
                    "rating": "Rating",
                    "num_reviews": "Reviews",
                    "price": "Price ($)"
                }).style.format({
                    "Rating": "{:.2f} ★",
                    "Reviews": "{:,}",
                    "Price ($)": "${:.2f}"
                }),
                use_container_width=True
            )
        else:
            st.info("No products match criteria.")


# --- TAB 4: REVIEW SENTIMENT ANALYSIS ---
with tab_sentiment:
    st.subheader("💬 Customer Review Sentiment Analysis")
    st.caption("Natural language processing of customer feedback without paid API keys.")

    sentiment_engine = get_sentiment_engine()
    st.info(f"⚡ **Active Sentiment Engine:** `{sentiment_engine.engine_name}`")

    # Perform Sentiment Scoring
    df_with_sentiment = sentiment_engine.analyze_dataframe(active_df)

    sent_col1, sent_col2 = st.columns([1, 1.2])

    with sent_col1:
        fig_sent = plot_sentiment_breakdown(df_with_sentiment)
        st.pyplot(fig_sent, use_container_width=True)

    with sent_col2:
        st.markdown("#### 🎯 Star-Rating vs. Review Sentiment Alignment")
        st.caption("Detects whether textual review sentiment matches numerical star rating.")

        alignment_counts = df_with_sentiment["rating_sentiment_alignment"].value_counts().reset_index()
        alignment_counts.columns = ["Alignment Status", "Product Count"]
        st.dataframe(alignment_counts, use_container_width=True)

        st.markdown("#### 🔤 Informative Sentiment Keywords")
        top_pos_words = extract_key_terms(df_with_sentiment, label="Positive", top_n=6)
        top_neg_words = extract_key_terms(df_with_sentiment, label="Negative", top_n=6)

        kword_col1, kword_col2 = st.columns(2)
        with kword_col1:
            st.markdown("**Top Positive Terms:**")
            if top_pos_words:
                for w, c in top_pos_words:
                    st.markdown(f"- **{w}** (`{c}` occurrences)")
            else:
                st.caption("No positive terms found.")

        with kword_col2:
            st.markdown("**Top Critical Terms:**")
            if top_neg_words:
                for w, c in top_neg_words:
                    st.markdown(f"- **{w}** (`{c}` occurrences)")
            else:
                st.caption("No critical terms found.")

    st.markdown("---")
    st.subheader("⚠️ Academic Notice: Limitations of Lexicon Sentiment Analysis")
    st.markdown(
        """
        As part of standard Data Analytics methodology, rule-based lexicon techniques (e.g. VADER)
        provide interpretable results without requiring external API tokens, but inherit distinct theoretical limitations:
        """
    )
    for lim in get_sentiment_limitations():
        with st.expander(f"📌 {lim['limitation']}", expanded=False):
            st.write(lim["explanation"])


# --- TAB 5: INTERACTIVE DATA EXPLORER ---
with tab_explorer:
    st.subheader("🔍 Interactive Catalog Data Explorer")
    st.caption("Search, inspect, and export clean data records.")

    search_query = st.text_input(
        "🔎 Search by Product Name or Review Text:",
        value="",
        placeholder="Type keyword e.g., 'Wireless', 'Coffee', 'Battery'..."
    )

    filtered_explorer_df = df_with_sentiment if "df_with_sentiment" in locals() else active_df

    if search_query.strip():
        mask = (
            filtered_explorer_df["product_name"].str.contains(search_query, case=False, na=False) |
            filtered_explorer_df["review_text"].str.contains(search_query, case=False, na=False)
        )
        filtered_explorer_df = filtered_explorer_df[mask]

    st.write(f"Showing **{len(filtered_explorer_df)}** matching records:")
    st.dataframe(
        filtered_explorer_df[[
            "product_id", "product_name", "category", "rating",
            "num_reviews", "price", "review_text", "sentiment_label"
        ]].rename(columns={
            "product_id": "Product ID",
            "product_name": "Product Title",
            "category": "Category",
            "rating": "Rating",
            "num_reviews": "Reviews",
            "price": "Price ($)",
            "review_text": "Customer Review",
            "sentiment_label": "Sentiment"
        }),
        use_container_width=True
    )

    # CSV Download Button
    csv_bytes = filtered_explorer_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_product_analysis.csv",
        mime="text/csv",
        help="Export the active filtered dataset for offline spreadsheet analysis."
    )


# --- TAB 6: PROJECT & VIVA GUIDE ---
with tab_viva:
    st.subheader("🎓 College Minor Project & Viva Voce Guide")
    st.markdown(
        """
        This documentation is designed to assist college students during academic presentations,
        demonstrations, and viva examinations.
        """
    )

    with st.expander("🎯 1. Project Motivation & Problem Statement", expanded=True):
        st.markdown(
            """
            - **Problem Statement:** In contemporary e-commerce marketplaces, aggregate consumer feedback
              (star ratings, review volume, and textual sentiment) represents critical business intelligence.
              Unstructured data and noise make manual analysis infeasible.
            - **Solution:** A complete, automated end-to-end Data Analytics pipeline that ingests catalog data,
              purges anomalies (duplicates, nulls, out-of-range ratings), calculates statistical KPIs,
              and visually correlates product metrics with textual customer sentiment.
            """
        )

    with st.expander("⚙️ 2. Architectural & Data Pipeline Workflow", expanded=True):
        st.markdown(
            r"""
            1. **Ingestion Layer (`src/data_loader.py`):** Ingests local or user-uploaded CSVs, normalizes heterogeneous column headers.
            2. **Data Hygiene & Validation:** Applies boundary validation ($1.0 \le \text{rating} \le 5.0$), deduplicates records, casts numeric types safely.
            3. **Analytics Engine (`src/analytics.py`):** Calculates portfolio KPIs, category-level aggregations, and Pearson correlations ($r$).
            4. **Sentiment Processing (`src/sentiment.py`):** Uses NLTK VADER with standalone lexicon fallback to compute valence polarity.
            5. **Presentation Layer (`app.py` & `src/visualizer.py`):** Renders interactive charts and dynamic KPI cards via Streamlit.
            """
        )

    with st.expander("❓ 3. Frequently Asked Viva Voce Questions & Answers", expanded=False):
        st.markdown(
            r"""
            **Q1: Why did you choose Streamlit instead of Flask or Django?**  
            *Answer:* Streamlit is purpose-built for data science and analytics. It provides reactive caching (`@st.cache_data`), native support for Pandas DataFrames, and direct rendering of Matplotlib/Seaborn figures with minimal boilerplate, allowing faster prototyping and clean UI for analytics dashboards.

            **Q2: How does the application handle duplicate and corrupted data?**  
            *Answer:* `src/data_loader.py` validates schema columns, removes duplicates based on `product_id`, enforces numeric bounds ($1.0 \le \text{rating} \le 5.0$, non-negative price and review counts), and records all cleaning statistics for transparency in the Data Cleaning Report.

            **Q3: What algorithm is used for sentiment analysis and why not a paid API?**  
            *Answer:* We implemented NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) with a local rule-based fallback. VADER is specifically tuned for social media and customer reviews, requires zero paid API keys, runs locally offline, and provides full privacy and reproducibility.

            **Q4: How do you identify rating-sentiment discrepancies?**  
            *Answer:* We cross-reference numerical star ratings with textual sentiment polarity. A product with a rating $\ge 4.0$ but negative sentiment (or $\le 2.5$ with positive sentiment) is flagged as a discrepancy, helping merchants detect misleading ratings or courier shipping complaints.
            """
        )


# --- 8. Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 13px; padding-bottom: 20px;">
        <strong>Product Rating Analysis using Data Analytics</strong> • Built with Python, Pandas, Matplotlib, Seaborn & Streamlit • Offline & Open-Source
    </div>
    """,
    unsafe_allow_html=True
)
