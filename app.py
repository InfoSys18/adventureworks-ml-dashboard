import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from plotly.graph_objs import Margin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# --- ІМПОРТИ ДЛЯ КЛАСТЕРИЗАЦІЇ КЛІЄНТІВ ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# Налаштування сторінки
# --------------------------------------------------
st.set_page_config(
    page_title="AdventureWorks Sales Intelligence", page_icon="📊", layout="wide"
)

# # Стилізація для зменшення висоти контейнерів Plotly
# st.markdown(
#     """
#     <style>
#     .stPlotlyChart { height: 350px !important; }
#     </style>
# """,
#     unsafe_allow_html=True,
# )


# --------------------------------------------------
# Підключення до SQL Server
# --------------------------------------------------
@st.cache_resource
def get_connection():
    return st.connection("sqlserver", type="sql")


# --------------------------------------------------
# Завантаження даних
# --------------------------------------------------
@st.cache_data
def load_data():
    connection = get_connection()
    sql_path = Path(__file__).parent / "sql" / "01_sales_query.sql"
    query = sql_path.read_text(encoding="utf-8")
    df = connection.query(query)
    return df


# --------------------------------------------------
# Підготовка даних
# --------------------------------------------------
def prepare_data(df):
    df = df.copy()
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["SalesAmount"] = df["LineTotal"]
    df["CostAmount"] = df["OrderQty"] * df["StandardCost"]
    df["ProfitAmount"] = df["SalesAmount"] - df["CostAmount"]
    df["ProfitMargin"] = (
        df["ProfitAmount"]
        .div(df["SalesAmount"])
        .where(df["SalesAmount"] != 0, 0)
    )

    df["Channel"] = df["OnlineOrderFlag"].map(
        {
            '0': "Офлайн (Дилери)",
            '1': "Онлайн-магазин",
            False: "Офлайн (Дилери)",
            True: "Онлайн-магазин",
        }
    )
    return df
# --------------------------------------------------
# Завантаження початкових даних
# --------------------------------------------------
df_raw = load_data()
df = prepare_data(df_raw)
# --------------------------------------------------
# Бічна панель (Sidebar) — Фільтри
# --------------------------------------------------
st.sidebar.header("🎛 Фільтри")

# --- 1. Фільтр дат ---
min_date = df["OrderDate"].min().date()
max_date = df["OrderDate"].max().date()

# noinspection bad-argument-type
date_range = st.sidebar.date_input(
    'Період замовлень',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[
        (df["OrderDate"].dt.date >= start_date)
        & (df["OrderDate"].dt.date <= end_date)
    ]

# --- 2. Новий фільтр категорій ---
st.sidebar.markdown("---")  # Візуальна лінія-розділювач

# Беремо унікальні категорії із вже відфільтрованого за датами датасету
available_categories = sorted(df["Category"].dropna().unique())

selected_categories = st.sidebar.multiselect(
    "Категорії товарів",
    options=available_categories,
    default=available_categories,  # За замовчуванням обрано ВСІ категорії
    placeholder="Оберіть категорії...",
)

# Застосовуємо фільтр категорій до даних
if selected_categories:
    df = df[df["Category"].isin(selected_categories)]
else:
    # Якщо користувач зняв усі галочки — показуємо попередження і зупиняємо рендеринг графіків
    st.sidebar.warning("⚠️ Будь ласка, оберіть хоча б одну категорію.")
    st.warning("⚠️ Оберіть категорії на бічній панелі для відображення аналітики.")
    st.stop()

# --- 3. Новий фільтр каналу продажів ---
st.sidebar.markdown("---")

channel_options = ["Всі канали", "Онлайн-магазин", "Офлайн (Дилери)"]
selected_channel = st.sidebar.radio("Канал продажів", options=channel_options)

if selected_channel != "Всі канали":
    df = df[df["Channel"] == selected_channel]


# --------------------------------------------------
# Головний екран
# --------------------------------------------------
st.title("📊 AdventureWorks Sales Intelligence")
st.caption("Аналіз продажів на основі AdventureWorks2022")

# Перевірка на випадок, якщо фільтри видали пустий датасет
if df.empty:
    st.warning("⚠️ Немає даних за обраний період.")
    st.stop()

# --------------------------------------------------
# KPI Блоки
# --------------------------------------------------
total_sales = df["SalesAmount"].sum()
total_cost = df["CostAmount"].sum()
total_profit = df["ProfitAmount"].sum()
profit_margin = total_profit / total_sales if total_sales != 0 else 0

number_of_orders = df["SalesOrderID"].nunique()
number_of_customers = df["CustomerID"].nunique()
average_order_value = (
    (total_sales / number_of_orders) if number_of_orders != 0 else 0
)

# Пакуємо в компактні 4 колонки
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Profit Margin", f"{profit_margin:.1%}")
col4.metric("Avg Order Value", f"${average_order_value:,.0f}")

st.markdown("---")

# --------------------------------------------------
# Блок 1: Динаміка та Категорії (Аналіз обсягів та Маржі через Вкладки)
# --------------------------------------------------
st.markdown("---")

# Створюємо ТРИ вкладки на всю ширину екрана (додаємо ML)
tab_volumes, tab_margin, tab_ml = st.tabs([
    "📊 Обсяги продажів та прибутку",
    "🎯 Аналіз маржинальності (Margin)",
    "🔮 Прогнозування продажів (ML)"
])

# --- ВКЛАДКА 1: ГРАФІКИ - ОБСЯГИ ---
with tab_volumes:
    col_vol1, col_vol2 = st.columns(2)

    with col_vol1:
        st.subheader("📈 Monthly Sales Trend")
        monthly = (
            df.set_index("OrderDate")
            .resample("ME")
            .agg(Sales=("SalesAmount", "sum"), Profit=("ProfitAmount", "sum"))
        )
        monthly.index = monthly.index.to_period("M").astype(str)
        monthly = monthly.reset_index()

        fig_trend = px.line(
            monthly,
            x="OrderDate",
            y=["Sales", "Profit"],
            labels={"value": "Amount ($)", "OrderDate": "Month"},
            color_discrete_sequence=["#1f77b4", "#2ca02c"],
        )
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_vol2:
        st.subheader("🍕 Sales & Profit by Category")
        category_sales = (
            df.groupby("Category", as_index=False)
            .agg(Sales=("SalesAmount", "sum"), Profit=("ProfitAmount", "sum"))
            .sort_values("Sales", ascending=True)
        )

        fig_cat = px.bar(
            category_sales,
            y="Category",
            x=["Sales", "Profit"],
            barmode="group",
            orientation="h",
            labels={"value": "Amount ($)"},
            color_discrete_sequence=["#1f77b4", "#9467bd"],
        )
        fig_cat.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# --- ВКЛАДКА 2: ГРАФІКИ - АНАЛІЗ МАРЖІ ---
with tab_margin:
    col_mar1, col_mar2 = st.columns(2)

    with col_mar1:
        st.subheader("📈 Щомісячна динаміка маржинальності")

        # Перераховуємо маржу відносно агрегованих сум за місяць
        monthly_margin = (
            df.set_index("OrderDate")
            .resample("ME")
            .agg(Sales=("SalesAmount", "sum"), Profit=("ProfitAmount", "sum"))
        )
        monthly_margin["Margin"] = (monthly_margin["Profit"] / monthly_margin["Sales"] * 100).fillna(0)
        monthly_margin.index = monthly_margin.index.to_period("M").astype(str)
        monthly_margin = monthly_margin.reset_index()

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_trend_margin = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend_margin.add_trace(
            go.Scatter(x=monthly_margin["OrderDate"], y=monthly_margin["Sales"], name="Продажі ($)",
                       line=dict(color="#1f77b4", width=3)),
            secondary_y=False,
        )
        fig_trend_margin.add_trace(
            go.Scatter(x=monthly_margin["OrderDate"], y=monthly_margin["Margin"], name="Маржа (%)",
                       line=dict(color="#d62728", width=3, dash="dash")),
            secondary_y=True,
        )
        fig_trend_margin.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig_trend_margin.update_yaxes(title_text="Сума замовлень ($)", secondary_y=False)
        fig_trend_margin.update_yaxes(title_text="Маржинальність (%)", ticksuffix="%", secondary_y=True)

        st.plotly_chart(fig_trend_margin, use_container_width=True)

    with col_mar2:
        st.subheader("🍕 Рейтинг категорій за рівнем Margin")

        category_analytics = (
            df.groupby("Category", as_index=False)
            .agg(Sales=("SalesAmount", "sum"), Profit=("ProfitAmount", "sum"))
        )
        category_analytics["ProfitMargin(%)"] = (
                    category_analytics["Profit"] / category_analytics["Sales"] * 100).fillna(0)
        category_analytics = category_analytics.sort_values("ProfitMargin(%)", ascending=True)

        fig_cat_margin = px.bar(
            category_analytics,
            y="Category",
            x="ProfitMargin(%)",
            orientation="h",
            text=category_analytics["ProfitMargin(%)"].map("{:.1f}%".format),
            labels={"ProfitMargin(%)": "Маржа (%)"},
            color="ProfitMargin(%)",
            color_continuous_scale="Reds"
        )
        fig_cat_margin.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        fig_cat_margin.update_traces(textposition="outside")

        st.plotly_chart(fig_cat_margin, use_container_width=True)
# --- ВКЛАДКА 3: ПРОГНОЗУВАННЯ ТА МАШИННЕ НАВЧАННЯ ---
with tab_ml:
# --------------------------------------------------
# Блок 1: Прогноз продажів товарів
# --------------------------------------------------
    st.subheader("🔮 Прогноз продажів на 3 місяці")
    st.caption(
        "Модель побудована за допомогою алгоритму експоненційного згладжування (Holt-Winters), враховуючи обрані фільтри в бічній панелі.")

    # 1. Підготовка даних для часового ряду
    # Групування продажів за місяцями для поточної відфільтрованої вибірки
    ts_data = (
        df.set_index("OrderDate")
        .resample("W")
        .agg(Sales=("SalesAmount", "sum"))
        # .agg(Margin=("MarginAmount", "mean"))
    )
    # Перевірка: чи достатньо даних для навчання моделі
    if len(ts_data) < 6:
        st.warning(
            "⚠️ Недостатньо історичних точок (місяців) за обраними фільтрами для побудови прогнозу. Будь ласка, розширте часовий період в бічній панелі.")
    else:
        # 2. Навчання ML-моделі (Holt's Linear Trend Model - ідеально для коротких рядів з трендом)
        try:
            model = ExponentialSmoothing(
                ts_data["Sales"],
                trend="add",
                seasonal=None,  # Сезонність вимикаємо, бо даних менше ніж за 3 повних роки
                initialization_method="estimated"
            )
            model_fit = model.fit()

            # Прогнозуємо на 3 кроки (місяці) вперед
            forecast_steps = 3
            forecast = model_fit.forecast(steps=forecast_steps)

            # 3. Створюємо майбутні дати для прогнозу
            future_dates = pd.date_range(
                start=ts_data.index[-1] + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq="ME"
            )

            # Історичні дані та прогноз в один датафрейм для графіка
            df_hist = pd.DataFrame({"Sales": ts_data["Sales"], "Тип": "Історія"})
            df_fore = pd.DataFrame({"Sales": forecast.values, "Тип": "Прогноз"}, index=future_dates)
            df_full = pd.concat([df_hist, df_fore]).reset_index().rename(columns={"index": "Дата"})

            # Перетворюємо дати у красивий текстовий формат для осі Х (наприклад, 2014-07)
            df_full["Дата_Стр"] = df_full["Дата"].dt.to_period("M").astype(str)

            # 4. Візуалізація результату через Plotly
            fig_ml = px.line(
                df_full,
                x="Дата_Стр",
                y="Sales",
                color="Тип",
                labels={"Sales": "Сума продажів ($)", "Дата_Стр": "Місяць", "Тип": "Дані"},
                color_discrete_map={"Історія": "#1f77b4", "Прогноз": "#ff7f0e"},
                # Синій для історії, помаранчевий для прогнозу
                title="Історичні продажі та математичний прогноз"
            )

            # Робимо лінію прогнозу пунктирною для наочності
            fig_ml.update_traces(
                patch={"line": {"dash": "dash"}},
                selector={"name": "Прогноз"}
            )

            fig_ml.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_ml, use_container_width=True)

            # 5. Виведемо точні цифри прогнозу в таблицю
            st.markdown("🎯 **Очікувані обсяги продажів на наступні 3 місяці:**")
            df_fore_table = df_fore.copy()
            df_fore_table.index = df_fore_table.index.to_period("M")
            st.dataframe(
                df_fore_table["Sales"].to_frame().style.format("${:,.0f}"),
                use_container_width=True
            )

        except Exception as e:
            st.error(
                f"Помилка під час розрахунку моделі. Можливо, дані занадто зашумлені або специфічні для цієї категорії. Технічна помилка: {e}"
            )
# --------------------------------------------------
# Блок 2: Top-10 товарів (Дві колонки, лінійні діаграми)
# --------------------------------------------------
st.markdown("---")
st.subheader("🏆 Top 10 Products Performance")

# Групуємо всі товари
top_products = (
    df.groupby("ProductName", as_index=False)
    .agg(Sales=("SalesAmount", "sum"), Profit=("ProfitAmount", "sum"))
)

# Окремо відбираємо ТОП-10 за продажами та ТОП-10 за прибутком
top10_sales = top_products.sort_values("Sales", ascending=True).tail(10)
top10_profit = top_products.sort_values("Profit", ascending=True).tail(10)

prod_col1, prod_col2 = st.columns(2)

with prod_col1:
    fig_prod_sales = px.bar(
        top10_sales,
        x="Sales",
        y="ProductName",
        orientation="h",
        color="Sales",
        color_continuous_scale="Blues",
        title="Top 10 Products by Sales",
    )
    fig_prod_sales.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False
    )
    st.plotly_chart(fig_prod_sales, use_container_width=True)

with prod_col2:
    fig_prod_profit = px.bar(
        top10_profit,
        x="Profit",
        y="ProductName",
        orientation="h",
        color="Profit",
        color_continuous_scale="Greens",
        title="Top 10 Products by Profit",
    )
    fig_prod_profit.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False
    )
    st.plotly_chart(fig_prod_profit, use_container_width=True)

# --------------------------------------------------
# Блок 3: Сегментація клієнтів (RFM + K-Means)
# --------------------------------------------------
st.markdown("---")
st.subheader("👥 Сегментація клієнтів за допомогою Machine Learning (K-Means)")
st.caption(
    "Аналіз поведінки покупців за трьома метриками: Свіжість (Recency), Частота (Frequency) та Гроші (Monetary).")

# 1. Розрахунок метрик RFM для кожного клієнта
max_date = df["OrderDate"].max()

rfm = (
    df.groupby("CustomerID")
    .agg(
        Recency=("OrderDate", lambda x: (max_date - x.max()).days),
        Frequency=("SalesOrderID", "nunique"),
        Monetary=("SalesAmount", "sum")
    )
    .reset_index()
)

# Перевірка на випадок, якщо клієнтів занадто мало для кластеризації
if len(rfm) < 4:
    st.info(
        "ℹ️ Недостатньо унікальних клієнтів у поточному зрізі даних для кластеризації. Спробуйте скинути фільтри або обрати ширший період.")
else:
    # 2. Інтерфейс: вибір кількості кластерів через радіокнопки
    col_input, col_space = st.columns([1, 2])
    with col_input:
        n_clusters = st.radio(
            "Оберіть кількість сегментів (кластерів):",
            options=[2, 3, 4, 5],
            index=1,  # За замовчуванням обрано 3 кластери
            horizontal=True
        )

    # 3. Підготовка даних (Масштабування)
    # K-Means чутливий до масштабу, тому StandardScaler обов'язковий
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # 4. Навчання моделі K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Перетворюємо номер кластера на красивий текст для легенди графіку
    rfm["Сегмент"] = rfm["Cluster"].apply(lambda x: f"Сегмент {x + 1}")

    # 5. Візуалізація результатів (3D діаграма Plotly)
    fig_3d = px.scatter_3d(
        rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Сегмент",
        opacity=0.7,
        log_z=True,
        labels={
            "Recency": "Свіжість (днів)",
            "Frequency": "Частота замовлень",
            "Monetary": "Гроші ($)"
        },
        title=f"Розподіл клієнтів на {n_clusters} сегменти у просторі RFM",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Налаштування макета
    fig_3d.update_layout(
        height=800,  # <-- Встановлюємо 800px висоти без перешкод з боку CSS
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        scene=dict(
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3)  # Трохи змістили камеру для кращого кута огляду
            ),
            aspectmode="cube",
            xaxis=dict(title=dict(text="Свіжість (днів з ост. покупки)", font=dict(size=11)), tickfont=dict(size=9)),
            yaxis=dict(title=dict(text="Частота (к-сть замовлень)", font=dict(size=11)), tickfont=dict(size=9)),
            zaxis=dict(title=dict(text="Гроші (обсяг продажів, $)", font=dict(size=11)), tickfont=dict(size=9))
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # 6. Статистика по кластерах (Середні значення метрик)
    st.markdown("📋 **Середні показники для кожного отриманого сегмента:**")

    cluster_stats = (
        rfm.groupby("Сегмент")
        .agg(
            Клієнтів=("CustomerID", "count"),
            Середня_Свіжість_Днів=("Recency", "mean"),
            Середня_Частота_Замовлень=("Frequency", "mean"),
            Загальні_Продажі_USD=("Monetary", "sum"),
            Середній_Чек_USD=("Monetary", "mean")
        )
        .round(1)
    )

    # Форматуємо таблицю для гарного бізнес-вигляду
    st.dataframe(
        cluster_stats.style.format({
            "Клієнтів": "{:,}",
            "Середня_Свіжість_Днів": "{:.0f} дн.",
            "Середня_Частота_Замовлень": "{:.1f} разів",
            "Загальні_Продажі_USD": "${:,.0f}",
            "Середній_Чек_USD": "${:,.0f}"
        }),
        use_container_width=True
    )
# --------------------------------------------------
# Детальні дані
# --------------------------------------------------
st.markdown("---")
with st.expander(f"📦 Переглянути сирі дані за обраний період ({len(df):,} рядків)"):
    st.dataframe(df, use_container_width=True)
