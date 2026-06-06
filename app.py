import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

model = joblib.load("churn_model.pkl")
df = pd.read_csv("European_Bank CSV.csv")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("🏦 Customer Churn Prediction App")

st.sidebar.header("Customer Information")

year = st.sidebar.number_input("Year", 2025, 2030, 2025)

customer_id = st.sidebar.number_input(
    "Customer ID",
    min_value=1000000,
    max_value=999999999,
    value=15634602
)

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)

age = st.sidebar.number_input("Age", 18, 100, 35)

tenure = st.sidebar.number_input("Tenure", 0, 10, 5)

balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)

products = st.sidebar.number_input("Num Of Products", 1, 4, 1)

has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])

active_member = st.sidebar.selectbox("Is Active Member", [0, 1])

salary = st.sidebar.number_input(
    "Estimated Salary",
    0.0,
    300000.0,
    100000.0
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

st.write("### Enter customer details and click Predict")

if st.button("Predict Churn"):

    data = pd.DataFrame([[
        year,
        customer_id,
        credit_score,
        age,
        tenure,
        balance,
        products,
        has_card,
        active_member,
        salary,
        geo_germany,
        geo_spain,
        gender_male
    ]], columns=[
        'Year',
        'CustomerId',
        'CreditScore',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
        'Geography_Germany',
        'Geography_Spain',
        'Gender_Male'
    ])
    
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.write("Prediction Value:", prediction[0])

    probability = model.predict_proba(data_scaled)

    st.write("Probability of Staying:",
         round(probability[0][0] * 100, 2), "%")

    st.write("Probability of Churning:",
         round(probability[0][1] * 100, 2), "%")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")
st.markdown("---")
st.header("📊 Customer Churn Analytics Dashboard")

col1, col2, col3 = st.columns(3)

# Chart 1
with col1:
    st.subheader("Distribution")

    fig, ax = plt.subplots(figsize=(3,2))

    df["Exited"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Churn")
    st.pyplot(fig)

# Chart 2
with col2:
    st.subheader("Geography")

    geo_churn = df.groupby("Geography")["Exited"].mean()

    fig, ax = plt.subplots(figsize=(3,2))

    geo_churn.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("By Geography")
    st.pyplot(fig)

# Chart 3
with col3:
    st.subheader("Gender")

    gender_churn = df.groupby("Gender")["Exited"].mean()

    fig, ax = plt.subplots(figsize=(3,2))

    gender_churn.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("By Gender")
    st.pyplot(fig)