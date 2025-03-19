import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج المدرب
model = joblib.load("ml_model/heart_disease_model.pkl")

# تحميل أسماء الأعمدة المستخدمة أثناء التدريب
df = pd.read_csv(r"C:\Users\user\Desktop\Project (1)intelligent prog\cleaned_data.csv")
expected_columns = df.drop(columns=['target']).columns  # إزالة عمود التصنيف

# إنشاء واجهة المستخدم
st.title(" Heart Disease Prediction System")

age = st.slider("Age", 20, 80, 50)
cholesterol = st.slider("Cholesterol Level", 100, 300, 200)
blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])

# إنشاء DataFrame للمدخلات
user_input = pd.DataFrame([[age, cholesterol, blood_pressure, smoking == "Yes", exercise == "Yes"]],
                          columns=['age', 'cholesterol', 'blood_pressure', 'smoking', 'exercise'])

# إضافة الأعمدة الناقصة وتعيينها إلى 0
for col in expected_columns:
    if col not in user_input.columns:
        user_input[col] = 0  # القيم الافتراضية للأعمدة غير الموجودة

# إعادة ترتيب الأعمدة بنفس ترتيب التدريب
user_input = user_input[expected_columns]

# التنبؤ
prediction = model.predict(user_input)[0]

if st.button("Predict"):
    if prediction == 1:
        st.error(" High risk of heart disease!")
    else:
        st.success(" Low risk of heart disease!")
