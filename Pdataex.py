import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# تحميل البيانات
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Survived', 'Pclass', 'Sex', 'Age']]  # اختيار الأعمدة الضرورية
    df = df.dropna()  # إزالة الصفوف التي تحتوي على قيم مفقودة
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # تحويل الجنس إلى قيم عددية
    return df

# إعداد واجهة Streamlit
st.title("Titanic Survival Prediction Models")
st.write("توقع ما إذا كان الركاب قد نجو من حادث سفينة تايتانيك باستخدام عدة نماذج تصنيفية.")

# تحديد مسار الملف بشكل ثابت
file_path = r"C:\Users\Abu8d\OneDrive\سطح المكتب\مشروع استخراج البيانات\Titanic-Dataset.csv"

# تحميل البيانات
df = load_data(file_path)

# تقسيم البيانات إلى المدخلات (features) والهدف (target)
X = df[['Pclass', 'Sex', 'Age']]  # المدخلات
y = df['Survived']  # الهدف (هل نجا الراكب؟)

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مقياس التوحيد (Standardization) للبيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 1. Logistic Regression
# ========================================
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# عرض نتائج النموذج Logistic Regression
st.header("Logistic Regression Results")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred_log_reg):.4f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_log_reg))

# ========================================
# 2. Decision Tree
# ========================================
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# عرض نتائج نموذج Decision Tree
st.header("Decision Tree Results")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred_tree):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred_tree):.4f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred_tree):.4f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred_tree):.4f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_tree))

# ========================================
# 3. Random Forest
# ========================================
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# عرض نتائج نموذج Random Forest
st.header("Random Forest Results")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred_rf):.4f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_rf))

# ========================================
# 4. Naive Bayes
# ========================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# عرض نتائج نموذج Naive Bayes
st.header("Naive Bayes Results")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred_nb):.4f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred_nb):.4f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred_nb):.4f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_nb))

# ========================================
# 5. User Input for Prediction
# ========================================
st.header("تنبؤ باستخدام نموذج Logistic Regression")

# إدخال بيانات جديدة من المستخدم
pclass = st.selectbox("Pclass (1, 2, or 3):", [1, 2, 3])
sex = st.selectbox("Sex (Male: 0, Female: 1):", [0, 1])
age = st.number_input("Age:", min_value=0, max_value=100, value=25)

# تحضير البيانات المدخلة للتنبؤ
user_data = pd.DataFrame([[pclass, sex, age]], columns=['Pclass', 'Sex', 'Age'])

# توحيد البيانات المدخلة
user_data_scaled = scaler.transform(user_data)

# التنبؤ باستخدام النموذج المدرب
user_prediction = log_reg_model.predict(user_data_scaled)

# عرض التنبؤ للمستخدم
if user_prediction[0] == 1:
    st.write("الراكب **نجا** من الحادث.")
else:
    st.write("الراكب **لم ينجو** من الحادث.")
