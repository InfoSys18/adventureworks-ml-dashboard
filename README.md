# 📊 AdventureWorks Sales Intelligence & ML Dashboard

An interactive Business Intelligence and Machine Learning web application built with **Python**, **Streamlit**, and **Plotly**, integrated with the **AdventureWorks2022** MS SQL database.

## 🔗 Live Demo
Check out the live interactive dashboard here: 
👉 [AdventureWorks ML Dashboard](https://adventureworks-ml-dashboard-rnthcufm2f5eiszktkutv7.streamlit.app/)

## 🚀 Features

- **Dynamic BI Analytics:** Multi-tab interface for tracking Sales, Profit, and Profit Margin performance.
- **Advanced Filtering:** Reactive sidebar to filter data by date range, product categories, and sales channels (Online vs. Reseller).
- **Time Series Forecasting (ML):** Built-in **Holt-Winters Exponential Smoothing** model to forecast sales 3 months ahead for each product category individually.
- **Customer Segmentation (ML):** **RFM Analysis** coupled with an interactive **3D K-Means Clustering** visualization, allowing dynamic selection of cluster counts.

## 🛠️ Tech Stack

- **Backend/Logic:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-learn (K-Means), Statsmodels (Exponential Smoothing)
- **Data Visualization:** Plotly Express, Plotly Graph Objects
- **Web Framework:** Streamlit
- **Database Connection:** SQL Server (SQLAlchemy / DBAPI)

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/InfoSys18/adventureworks-ml-dashboard
   cd adventureworks-ml-dashboard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
