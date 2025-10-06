import streamlit as st
import pandas as pd
import cloudpickle

# -----------------------------
# Load saved models
# -----------------------------
with open(r"C:\MultipleDiseasePrediction\env\Scripts\kidney_disease_pipeline.pkl", "rb") as f:
    kidney_model = cloudpickle.load(f)

with open(r"C:\MultipleDiseasePrediction\env\Scripts\liver_disease_pipeline.pkl", "rb") as f:
    liver_model = cloudpickle.load(f)

with open(r"C:\MultipleDiseasePrediction\env\Scripts\parkinsons_xgb_pipeline.pkl", "rb") as f:
    parkinsons_model = cloudpickle.load(f)

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Multiple Disease Prediction System")
page = st.sidebar.radio("Navigation", 
                        ["Kidney Prediction", "Liver Prediction", "Parkinson's Prediction"])

# -----------------------------
# Kidney Disease Prediction Page (Top 10 Features)
# -----------------------------
if page == "Kidney Prediction":
    st.title("Kidney Disease Prediction")

    # Categorical features
    dm = st.selectbox("Diabetes Mellitus (dm)", ["No", "Yes"])
    pe = st.selectbox("Pedal Edema (pe)", ["No", "Yes"])
    htn = st.selectbox("Hypertension (htn)", ["No", "Yes"])

    # Numeric features
    hemo = st.number_input("Hemoglobin (hemo)", value=0.0, step=0.1, format="%.2f")
    sg = st.number_input("Specific Gravity (sg)", value=0.0, step=0.01, format="%.2f")
    al = st.number_input("Albumin (al)", value=0.0, step=0.01, format="%.2f")
    sc = st.number_input("Serum Creatinine (sc)", value=0.0, step=0.01, format="%.2f")
    bu = st.number_input("Blood Urea (bu)", value=0.0, step=0.01, format="%.2f")
    pc = st.selectbox("Pus Cell (pc)", ["Normal", "Abnormal"])

    # Convert categorical to numeric/string as in CSV
    dm_val = "yes" if dm == "Yes" else "no"
    pe_val = "yes" if pe == "Yes" else "no"
    htn_val = "yes" if htn == "Yes" else "no"
    pc_val = pc.lower()  # match CSV ("normal"/"abnormal")

    if st.button("Kidney Disease Test Result"):
        features = pd.DataFrame([[dm_val, pe_val, hemo, htn_val, sg, al, sc, bu, pc_val]],
                                columns=['dm','pe','hemo','htn','sg','al','sc','bu','pc'])
        prediction = kidney_model.predict(features)[0]
        if prediction == 1:
            st.error("⚠️ The person has Kidney Disease")
        else:
            st.success("✅ The person does not have Kidney Disease")

# -----------------------------
# Liver Disease Prediction Page
# -----------------------------
elif page == "Liver Prediction":
    st.title("Liver Disease Prediction")

    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    alkphos = st.number_input("Alkaline Phosphotase")
    sgpt = st.number_input("Alamine Aminotransferase")
    sgot = st.number_input("Aspartate Aminotransferase")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("Albumin")
    agr = st.number_input("Albumin and Globulin Ratio")

    gender_num = 1 if gender == "Male" else 0

    if st.button("Liver Disease Test Result"):
        features = pd.DataFrame([[age, gender_num, tb, db, alkphos, sgpt, sgot, tp, alb, agr]],
                                columns=["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                                         "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                                         "Aspartate_Aminotransferase", "Total_Protiens",
                                         "Albumin", "Albumin_and_Globulin_Ratio"])
        prediction = liver_model.predict(features)[0]
        if prediction == 1:
            st.error("⚠️ The person has Liver Disease")
        else:
            st.success("✅ The person does not have Liver Disease")

# -----------------------------
# Parkinson’s Disease Prediction Page (Top 15 Features)
# -----------------------------
elif page == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")

    PPE = st.number_input("PPE", format="%.5f")
    spread1 = st.number_input("spread1", format="%.5f")
    fo = st.number_input("MDVP:Fo(Hz)", format="%.5f")
    spread2 = st.number_input("spread2", format="%.5f")
    flo = st.number_input("MDVP:Flo(Hz)", format="%.5f")
    fhi = st.number_input("MDVP:Fhi(Hz)", format="%.5f")
    ddp = st.number_input("Jitter:DDP", format="%.5f")
    nhr = st.number_input("NHR", format="%.5f")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", format="%.5f")
    apq5 = st.number_input("Shimmer:APQ5", format="%.5f")
    rpde = st.number_input("RPDE", format="%.5f")
    shimmer = st.number_input("MDVP:Shimmer", format="%.5f")
    dfa = st.number_input("DFA", format="%.5f")
    hnr = st.number_input("HNR", format="%.5f")
    rap = st.number_input("MDVP:RAP", format="%.5f")

    if st.button("Parkinson's Test Result"):
        features = pd.DataFrame([[PPE, spread1, fo, spread2, flo, fhi, ddp, nhr,
                                  jitter_abs, apq5, rpde, shimmer, dfa, hnr, rap]],
                                columns=['PPE','spread1','MDVP:Fo(Hz)','spread2','MDVP:Flo(Hz)',
                                         'MDVP:Fhi(Hz)','Jitter:DDP','NHR','MDVP:Jitter(Abs)',
                                         'Shimmer:APQ5','RPDE','MDVP:Shimmer','DFA','HNR','MDVP:RAP'])
        prediction = parkinsons_model.predict(features)[0]
        if prediction == 1:
            st.error("⚠️ The person is likely to have Parkinson's Disease.")
        else:
            st.success("✅ The person is unlikely to have Parkinson's Disease.")
