import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# تحميل البيانات
df = pd.read_csv(r"C:\Users\user\Desktop\Project (1)intelligent prog\cleaned_data.csv")

# فصل الميزات عن الهدف
X = df.drop(columns=["target"])
y = df["target"]

# تدريب النموذج
model = DecisionTreeClassifier()
model.fit(X, y)

# إنشاء المجلد إذا لم يكن موجودًا
import os
if not os.path.exists("ml_model"):
    os.makedirs("ml_model")

# حفظ النموذج
joblib.dump(model, "ml_model/heart_disease_model.pkl")
print("✅ Model saved successfully in ml_model/heart_disease_model.pkl")
