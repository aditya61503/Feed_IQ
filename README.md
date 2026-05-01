# ML Feedback Intelligence 🧠

A Streamlit-based application that uses Machine Learning (TF-IDF + K-Means) to analyze, cluster, and prioritize user feedback automatically.

## 🚀 Features

- **Automated Clustering**: Groups similar feedback into themes (e.g., "Pricing Issues", "Feature Requests") using K-Means.
- **Priority Scoring**: Auto-calculates priority (High/Medium/Low) based on similarity to other feedback (impact) and sentiment.
- **Smart Tagging**: Extracts key keywords from feedback text.
- **CSV Model**:
    - **Upload your own data**: Supports drag-and-drop CSV upload.
    - **Aggressive Cleaning**: Automatically strips numbers/symbols to focus on text.
    - **Header Detection**: Handles headerless CSVs and auto-detects column names.
- **Neutral UI**: Clean, professional interface with a focus on data readability.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aditya61503/Feed_IQ.git
    cd ML_Feedback_Intelligence
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  Run the application:
    ```bash
    streamlit run app_fixed.py
    ```

2.  **Upload Data**:
    - Use the sidebar "📂 Data Source" to upload a CSV file.
    - The app will automatically find the text column and analyze it.

3.  **Analyze**:
    - View **Priority Issues** to see what needs immediate attention.
    - Check **Categories** to see the distribution of feedback themes.
    - Use **Similar Finder** to group duplicate reports.

## 📂 Project Structure

- `app_fixed.py`: Main Streamlit application frontend.
- `ml_engine.py`: Core Logic for text vectorization, clustering, and scoring.
- `data_manager_fixed.py`: Handles data loading and saving.
- `dataset.csv`: Default sample dataset.

## 🤖 Tech Stack

- **Python**: Core language.
- **Streamlit**: Web interface.
- **Scikit-learn**: TF-IDF and K-Means clustering.
- **Pandas/Numpy**: Data manipulation.



## 📸 Screenshots

### 🖥️ Dashboard
![Dashboard](screenshots/Dashboard.png)

### 📊 Clustering Output
![Clusters](screenshots/Clusters.png)

### 🔍 Similar Feedback Detection
![Similar Finds](screenshots/Similar_finds.png)
---
*Created by Aditya Walse-Patil*
