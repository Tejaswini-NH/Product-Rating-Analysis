# Product Rating Analysis using Data Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-blueviolet.svg)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end exploratory data analytics and customer review intelligence dashboard built for **College Minor Projects**, **Academic Capstones**, and **Data Analytics Practicums**.

---

## 🌟 Key Features

1. **Interactive Web Dashboard**: Built with Streamlit, featuring a modern responsive interface with metric cards and custom CSS.
2. **Dual Data Ingestion**:
   - Ships with an instant-run curated dataset (`data/sample_products.csv`) spanning 6 major categories.
   - Dynamic **CSV File Uploader** to analyze custom e-commerce product files.
3. **Automated Data Hygiene & Cleaning**:
   - Deduplication by product identifier and name.
   - Missing value imputation and null purging.
   - Boundary validation ($1.0 \le \text{rating} \le 5.0$, non-negative price, non-negative reviews).
   - Live **Data Cleaning Report** badge showing cleaning statistics.
4. **Business Intelligence & KPIs**:
   - Instant calculation of Total Products, Mean Rating, Total Reviews, Low-Rated Percentage ($< 3.0$ stars), and Average Price.
   - Category-level breakdown (ratings, engagement volume, pricing distribution).
5. **Exploratory Visualizations**:
   - Rating distribution histogram with KDE curve and mean/median markers.
   - Horizontal category rating comparison with target quality benchmarks.
   - Review volume distribution bar chart.
   - Price vs. Rating scatter plot with regression trendline.
   - Subplots comparing Top 5 Best-Rated vs. Bottom 5 Underperforming products.
6. **Customer Review Sentiment Analysis**:
   - Lexicon-based sentiment scoring (NLTK VADER with a zero-dependency fallback lexicon).
   - Categorization into Positive, Neutral, and Negative sentiments.
   - Cross-referencing star ratings against textual sentiment to detect **rating-sentiment discrepancies**.
   - Top positive and negative review keyword extraction.
   - Explicit academic disclosure of **Methodological Limitations**.
7. **Dynamic Data-Driven Insights**:
   - Automatically generated insights based strictly on active catalog numbers (no hallucinated text).
8. **Interactive Data Explorer**:
   - Searchable, sortable, and filterable catalog table with one-click **CSV Export**.
9. **Viva Voce & College Project Guide**:
   - Integrated tab with architecture explanation, problem statement, and examiner Q&A.

---

## 🏗️ Project Architecture & File Structure

```
Product-Rating-Analysis/
│
├── data/
│   └── sample_products.csv        # Realistic dataset across 6 product categories
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # CSV ingestion, normalization, and validation pipeline
│   ├── analytics.py               # KPI formulas, aggregations, correlations & insights
│   ├── sentiment.py               # VADER & rule-based sentiment engine with term extraction
│   └── visualizer.py              # Publication-grade Matplotlib & Seaborn chart builders
│
├── tests/
│   ├── __init__.py                # Test package initializer
│   └── test_pipeline.py           # Automated unit & integration tests
│
├── app.py                         # Main Streamlit web application
├── requirements.txt               # Project dependencies (zero paid APIs required)
├── .gitignore                     # Git ignore rules for Python & Streamlit
└── README.md                      # Complete documentation & viva guide
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed:
```bash
python --version
```

### 2. Clone or Navigate to the Project Directory
```bash
cd path/to/Product-Rating-Analysis
```

### 3. Create and Activate a Virtual Environment (Recommended)
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download NLTK VADER Lexicon (One-time setup)
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```
*(Note: If you run offline, the built-in fallback lexicon automatically activates seamlessly without errors.)*

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Tests

To verify data cleaning, KPI metrics calculation, sentiment analysis, and visualizations:
```bash
python -m unittest discover -s tests -v
```

**Expected output:**
```
test_01_data_loading_and_cleaning ... ok
test_02_kpi_metrics_calculation ... ok
test_03_category_analytics ... ok
test_04_product_rankings ... ok
test_05_correlation_and_insights ... ok
test_06_sentiment_analysis ... ok
test_07_visualizations_generation ... ok
----------------------------------------------------------------------
Ran 7 tests in ~0.7s

OK
```

---

## 📊 Dataset Schema

When uploading custom CSV files, the system automatically detects standard column variations:

| Canonical Column | Accepted Aliases | Description | Type / Range |
| :--- | :--- | :--- | :--- |
| `product_id` | `id`, `productid`, `sku`, `asin` | Unique product identifier | String |
| `product_name` | `title`, `name`, `item_name` | Title of the product | String (Mandatory) |
| `category` | `department`, `genre`, `type` | Product category | String |
| `rating` | `stars`, `score`, `overall_rating` | Customer rating | Float ($1.0$ to $5.0$) |
| `num_reviews` | `review_count`, `total_reviews` | Number of reviews | Integer ($\ge 0$) |
| `price` | `cost`, `unit_price`, `sale_price` | Product price | Float ($\ge 0$) |
| `review_text` | `review`, `comments`, `feedback` | Customer review text | Text (Optional) |

---

## 🎓 College Viva Voce & Presentation Cheat-Sheet

| Viva Question | Concise Answer |
| :--- | :--- |
| **What is the goal of this project?** | To turn raw e-commerce product catalog and review data into actionable business intelligence using data cleaning, statistical KPIs, visual distributions, and review sentiment mining. |
| **How did you handle dirty or missing data?** | `src/data_loader.py` enforces boundary checks ($1.0 \le \text{rating} \le 5.0$), eliminates duplicate IDs, drops critical nulls, and reports all cleaned anomalies in a dedicated hygiene summary. |
| **Why use NLTK VADER for sentiment analysis?** | VADER is a rule-based lexicon specifically optimized for consumer reviews and short text. It runs entirely offline, requires **no paid API keys**, and gives transparent valence scores. |
| **What are the limitations of this sentiment model?** | Inability to detect sarcasm ("Great, it broke in 10 minutes"), sensitivity to complex negations, and lack of domain context (e.g. "heavy" being positive for gym weights but negative for phones). |
| **What does the Pearson correlation ($r$) measure here?** | It measures the linear relationship between Price and Rating (or Review Count and Rating). A value near 0 indicates that price does not dictate customer satisfaction. |

---

## 🐙 Step-by-Step GitHub Upload Guide

Follow these commands to push this project to your GitHub profile:

1. **Initialize Git repository:**
   ```bash
   git init
   ```

2. **Stage all files:**
   ```bash
   git add .
   ```

3. **Create the initial commit:**
   ```bash
   git commit -m "Initial commit: Complete Product Rating Analysis Minor Project"
   ```

4. **Rename the default branch to main:**
   ```bash
   git branch -M main
   ```

5. **Create a new repository on GitHub:**
   - Go to [GitHub](https://github.com/new).
   - Name your repository (e.g., `Product-Rating-Analysis`).
   - Do **not** initialize with a README (we already have a complete one).
   - Click **Create repository**.

6. **Link your local repository to GitHub:**
   ```bash
   git remote add origin https://github.com/<YOUR-USERNAME>/Product-Rating-Analysis.git
   ```

7. **Push your code to GitHub:**
   ```bash
   git push -u origin main
   ```

---

## 📝 License
This project is open-source and released under the [MIT License](LICENSE). Suitable for academic submissions and educational learning.
