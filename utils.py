import pandas as pd
import numpy as np

path = "C:/Users/PC/Desktop/tasas_mercado/" # NBA/ecommerce










"""
SELECCION DE VARIABLES
El código incluye:

✔ Limpieza
✔ Reducción por correlación
✔ Mutual Information
✔ ANOVA F-score
✔ Chi²
✔ Importancia con RandomForest
✔ Importancia con XGBoost
✔ Selección final
✔ Cálculo de MAP@K
✔ Ranking final de variables
"""


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# ============
# 1. CARGA Y PREPARACIÓN
# ============

df = df_original.copy()

TARGET = "producto_tomado"     # variable multinomial
y = df[TARGET]
X = df.drop(columns=[TARGET])

# Convertir el target a números
le = LabelEncoder()
y = le.fit_transform(y)

# Identificar variables numéricas / categóricas
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns

# ============
# 2. LIMPIEZA INICIAL
# ============

# Quitar columnas constantes
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=const_cols)

# Quitar columnas con >90% NA o ceros
high_null = [c for c in X.columns if X[c].isna().mean() > 0.90]
high_zero = [c for c in num_cols if (X[c] == 0).mean() > 0.90]

X = X.drop(columns=list(set(high_null + high_zero)))

# Imputación simple
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("NA")

# One-hot para categóricas
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Actualizamos las columnas numéricas
num_cols = X.columns

# ============
# 3. REDUCCIÓN POR CORRELACIÓN
# ============

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)

# ============
# 4. SELECCIÓN UNIVARIADA
# ============

print("Calculando Mutual Information...")
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
top_mi = mi_series.head(150).index.tolist()

print("Calculando ANOVA F-score...")
fvals = f_classif(X, y)[0]
f_series = pd.Series(fvals, index=X.columns).sort_values(ascending=False)
top_anova = f_series.head(80).index.tolist()

print("Calculando Chi²...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_vals, p_vals = chi2(X_scaled, y)
chi_series = pd.Series(chi_vals, index=X.columns).sort_values(ascending=False)
top_chi = chi_series.head(50).index.tolist()

# Unir variables semifinalistas
vars_semifinalists = list(set(top_mi + top_anova + top_chi))
X2 = X[vars_semifinalists]

# ============
# 5. MODEL-BASED SELECTION (RandomForest + XGBoost)
# ============

# ---- Random Forest ----
print("Entrenando RandomForest...")
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X2, y)
rf_imp = pd.Series(rf.feature_importances_, index=X2.columns).sort_values(ascending=False)
top_rf = rf_imp.head(100).index.tolist()

# ---- XGBoost Multinomial ----
print("Entrenando XGBoost...")
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
xgb.fit(X2, y)

xgb_imp = pd.Series(xgb.feature_importances_, index=X2.columns).sort_values(ascending=False)
top_xgb = xgb_imp.head(80).index.tolist()

# Unir finalistas
vars_final = list(set(top_rf + top_xgb))
X3 = X[vars_final]

print(f"Variables finales preliminares: {len(vars_final)}")

# ============
# 6. CALCULAR MAP@k
# ============

def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        p = p[:k]
        if a in p:
            score += 1 / (p.index(a) + 1)
    return score / len(actual)

def evaluar_MAP(model, X, y):
    proba = model.predict_proba(X)
    preds = np.argsort(-proba, axis=1).tolist()
    return {
        "MAP@1": mapk(y, preds, k=1),
        "MAP@2": mapk(y, preds, k=2),
        "MAP@3": mapk(y, preds, k=3),
        "MAP@4": mapk(y, preds, k=4),
    }

print("Evaluando MAP...")
scores = evaluar_MAP(xgb, X3, y)
print(scores)

# ============
# 7. SHAP PARA EXPLICABILIDAD
# ============

print("Calculando SHAP values (puede tardar)...")
explainer = shap.TreeExplainer(xgb)
sh_values = explainer.shap_values(X3)

shap_importance = pd.DataFrame({
    "variable": X3.columns,
    "shap_importance": np.mean(np.abs(sh_values), axis=1).mean(axis=0)
}).sort_values(by="shap_importance", ascending=False)

print("\nIMPORTANCIA SHAP TOP 30:")
print(shap_importance.head(30))

# ============
# 8. RESULTADO FINAL
# ============

vars_final_recomendadas = shap_importance.head(50)["variable"].tolist()

print("\n===== VARIABLES SELECCIONADAS FINALES =====")
for v in vars_final_recomendadas:
    print(v)
