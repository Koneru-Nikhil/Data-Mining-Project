import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# 🌍 Title
# -------------------------------
st.title("🌍 TripMine: Travel Destination Analytics & Demand Prediction")
st.markdown("### 📊 Data Warehousing & Data Mining Project")

# -------------------------------
# 📂 Load Dataset
# -------------------------------
try:
    df = pd.read_csv("Travel.csv")
except:
    st.error("❌ Travel.csv file not found. Please keep it in the same folder.")
    st.stop()

# -------------------------------
# 📌 1. Data Warehouse
# -------------------------------
st.header("📂 Data Warehouse (Centralized Data)")
st.write("Integrated travel data from multiple sources.")
st.dataframe(df.head())

# -------------------------------
# 📌 2. ETL Process
# -------------------------------
st.header("⚙ ETL Process")

st.subheader("🔍 Extract")
st.write("Shape of dataset:", df.shape)

st.subheader("🧹 Transform (Data Cleaning)")
st.write("Missing Values:")
st.write(df.isnull().sum())

clean_option = st.radio(
    "Choose Data Cleaning Method:",
    ("None", "Remove Missing Values", "Fill Missing Values (Mean)")
)

if clean_option == "Remove Missing Values":
    df = df.dropna()
    st.success("Missing values removed!")

elif clean_option == "Fill Missing Values (Mean)":
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    st.success("Missing values filled with mean!")

st.subheader("📥 Load")
st.write("Cleaned data ready for analysis")

# -------------------------------
# 📌 3. Popular Destination
# -------------------------------
st.header("📍 Popular Destination Analysis")

if "Destination" in df.columns:
    dest_count = df["Destination"].value_counts()
    st.bar_chart(dest_count)
    st.write("Top Destinations:")
    st.write(dest_count.head())
else:
    st.warning("Destination column not found!")

# -------------------------------
# 📌 4. Seasonal Trends
# -------------------------------
st.header("🌦 Seasonal Travel Trends")

if "Season" in df.columns:
    st.bar_chart(df["Season"].value_counts())
else:
    st.warning("Season column not found!")

# -------------------------------
# 📌 5. Budget Analysis
# -------------------------------
st.header("💰 Budget Analysis")

numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) > 0:
    col = st.selectbox("Select Numeric Column", numeric_cols)
    st.line_chart(df[col])
else:
    st.warning("No numeric columns available!")

# -------------------------------
# 📊 EXTRA VISUALIZATIONS
# -------------------------------

# Histogram
st.header("📊 Data Distribution (Histogram)")
if len(numeric_cols) > 0:
    col = st.selectbox("Select column for histogram", numeric_cols, key="hist")

    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

# Pie Chart
st.header("🥧 Destination Share")
if "Destination" in df.columns:
    dest_counts = df["Destination"].value_counts().head(5)

    fig, ax = plt.subplots()
    ax.pie(dest_counts, labels=dest_counts.index, autopct="%1.1f%%")
    ax.set_title("Top 5 Destinations Share")
    st.pyplot(fig)

# Box Plot
st.header("📦 Outlier Detection (Box Plot)")
if len(numeric_cols) > 0:
    col = st.selectbox("Select column for boxplot", numeric_cols, key="box")

    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna())
    ax.set_title(f"Box Plot of {col}")
    st.pyplot(fig)

# Correlation Heatmap
st.header("🔥 Correlation Heatmap")
corr = df.select_dtypes(include=np.number).corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr, cmap="coolwarm")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
st.pyplot(fig)

# Transport Analysis
st.header("🚗 Transport Mode Analysis")
if "Transport" in df.columns:
    st.bar_chart(df["Transport"].value_counts())

# Budget vs Destination
st.header("💰 Budget vs Destination")
if "Destination" in df.columns and len(numeric_cols) > 0:
    col = st.selectbox("Select budget column", numeric_cols, key="budget_dest")

    avg_budget = df.groupby("Destination")[col].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_budget)

# -------------------------------
# 📌 6. Clustering
# -------------------------------
st.header("🤖 Data Mining: Customer Segmentation")

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) >= 2:
    numeric_df = numeric_df.fillna(numeric_df.mean())

    k = st.slider("Select Number of Clusters", 2, 6, 3)

    if len(numeric_df) >= k:
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(numeric_df)

        df["Cluster"] = clusters

        st.write("Clustered Data:")
        st.dataframe(df[["Cluster"]].head())

        st.subheader("📍 Cluster Visualization")
        st.scatter_chart(
            numeric_df.iloc[:, :2].assign(cluster=clusters)
        )
    else:
        st.warning("Not enough data for clustering.")
else:
    st.warning("Need at least 2 numeric columns for clustering.")

# -------------------------------
# 📌 7. Demand Prediction
# -------------------------------
st.header("📈 Travel Demand Insight")

if "Destination" in df.columns:
    st.success(f"Most Demanded Destination: {df['Destination'].mode()[0]}")

# -------------------------------
# 📌 8. Conclusion
# -------------------------------
st.header("📌 Conclusion")

st.write("""
✔ Centralized Data Warehouse  
✔ ETL process implemented  
✔ Data Mining using K-Means  
✔ Multiple visualizations for insights  
✔ Identified popular destinations & trends  
✔ Helps in travel demand prediction  
""")

# Footer
st.markdown("---")
st.markdown("👨‍💻 TripMine Project | Data Warehousing & Mining")