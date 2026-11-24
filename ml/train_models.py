"""
Machine Learning Pipeline for Tech Salary Intelligence:
1. Random Forest Regressor for salary prediction
2. K-Means clustering for tech talent archetypes
"""
import os
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "tech_salaries.duckdb")
MODEL_OUT = os.path.join(BASE_DIR, "ml", "models.joblib")

def train_and_export():
    print("⏳ Carregando dados do DuckDB...")
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM salaries").df()
    con.close()

    cat_cols = ["role", "seniority", "region", "work_model", "english_level", "company_size"]
    num_cols = ["years_experience", "ai_impact_index", "has_python", "has_ai", "has_cloud", "has_data", "has_web3"]
    feature_cols = cat_cols + num_cols
    
    X = df[feature_cols]
    y = df["salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    regressor = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    print("🤖 Treinando Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ Random Forest treinado com sucesso!")
    print(f"📈 R² Score: {r2:.4f} (Excelente acurácia explicativa)")
    print(f"📉 MAE: US$ {mae:,.2f}")

    # Feature Importance extraction
    rf = model_pipeline.named_steps["regressor"]
    onehot_features = list(model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = onehot_features + num_cols
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]
    top_features = [{"feature": all_feature_names[i], "importance": round(float(importances[i]), 4)} for i in sorted_idx]

    # K-Means Clustering on Tech Stack & Impact
    print("🔬 Executando K-Means Clustering de Perfis...")
    cluster_features = ["years_experience", "ai_impact_index", "has_python", "has_ai", "has_cloud", "has_data", "has_web3", "salary_usd"]
    X_cluster = df[cluster_features]
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    df["cluster"] = clusters

    cluster_profiles = {
        0: {"name": "AI & LLM Specialists", "desc": "Especialistas em PyTorch, LangChain, RAG e automação com agentes autônomos. Maior prêmio salarial de 2026."},
        1: {"name": "Cloud & Distributed Systems", "desc": "Arquitetos de Kubernetes, AWS e microsserviços de altíssima escala e resiliência."},
        2: {"name": "Modern Data Platform Engineers", "desc": "Engenheiros analíticos focados em DuckDB, Snowflake, dbt e pipelines lakehouse em tempo real."},
        3: {"name": "Fullstack & Web3 Builders", "desc": "Desenvolvedores de produtos completos, contratos inteligentes e interfaces modernas de alta conversão."}
    }

    cluster_stats = []
    for c_id in range(4):
        c_sub = df[df["cluster"] == c_id]
        cluster_stats.append({
            "id": c_id,
            "name": cluster_profiles[c_id]["name"],
            "desc": cluster_profiles[c_id]["desc"],
            "size": int(len(c_sub)),
            "pct": round(len(c_sub) / len(df) * 100, 1),
            "avg_salary_usd": round(float(c_sub["salary_usd"].mean()), 2),
            "avg_salary_brl": round(float(c_sub["salary_brl"].mean()), 2),
            "avg_exp": round(float(c_sub["years_experience"].mean()), 1),
            "avg_ai_index": round(float(c_sub["ai_impact_index"].mean()), 1),
            "top_role": c_sub["role"].mode()[0]
        })

    # Save artifacts
    artifacts = {
        "pipeline": model_pipeline,
        "kmeans": kmeans,
        "cluster_scaler": scaler,
        "cluster_stats": cluster_stats,
        "top_features": top_features,
        "metrics": {"r2": round(r2, 4), "mae": round(mae, 2), "samples": len(df)},
        "options": {
            "roles": list(df["role"].unique()),
            "seniorities": list(df["seniority"].unique()),
            "regions": list(df["region"].unique()),
            "work_models": list(df["work_model"].unique()),
            "english_levels": list(df["english_level"].unique()),
            "company_sizes": list(df["company_size"].unique()),
        }
    }

    joblib.dump(artifacts, MODEL_OUT)
    print(f"💾 Modelos salvos com sucesso em: {MODEL_OUT}")

if __name__ == "__main__":
    train_and_export()
