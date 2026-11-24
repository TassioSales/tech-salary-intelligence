"""
FastAPI Backend for Tech Career & Global Remote Salary Intelligence.
Serves analytical endpoints and modern Tailwind/Chart.js interactive frontend.
"""
import os
import duckdb
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "tech_salaries.duckdb")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models.joblib")

app = FastAPI(title="Tech Career & Global Remote Salary Intelligence", version="1.0.0")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

# Load ML artifacts
artifacts = joblib.load(MODEL_PATH)
pipeline = artifacts["pipeline"]
kmeans = artifacts["kmeans"]
cluster_scaler = artifacts["cluster_scaler"]
cluster_stats = artifacts["cluster_stats"]
top_features = artifacts["top_features"]
metrics = artifacts["metrics"]
options = artifacts["options"]

class PredictRequest(BaseModel):
    role: str
    seniority: str
    region: str
    work_model: str
    english_level: str
    company_size: str
    years_experience: float
    has_python: int = 1
    has_ai: int = 0
    has_cloud: int = 0
    has_data: int = 0
    has_web3: int = 0

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "options": options,
        "metrics": metrics,
        "cluster_stats": cluster_stats,
        "top_features": top_features,
    })

@app.get("/api/analytics")
async def get_analytics():
    con = duckdb.connect(DB_PATH)
    
    # 1. Salary by Role
    role_df = con.execute("""
        SELECT role, round(avg(salary_usd), 0) as avg_usd, round(avg(salary_brl), 0) as avg_brl, count(*) as count
        FROM salaries GROUP BY role ORDER BY avg_usd DESC
    """).df().to_dict(orient="records")
    
    # 2. Salary by Seniority
    sen_df = con.execute("""
        SELECT seniority, round(avg(salary_usd), 0) as avg_usd, round(min(salary_usd), 0) as min_usd, round(max(salary_usd), 0) as max_usd
        FROM salaries GROUP BY seniority ORDER BY avg_usd ASC
    """).df().to_dict(orient="records")
    
    # 3. Salary by Region & Remote
    region_df = con.execute("""
        SELECT region, round(avg(salary_usd), 0) as avg_usd, count(*) as total
        FROM salaries GROUP BY region ORDER BY avg_usd DESC
    """).df().to_dict(orient="records")

    # 4. English Level Impact
    eng_df = con.execute("""
        SELECT english_level, round(avg(salary_usd), 0) as avg_usd
        FROM salaries GROUP BY english_level ORDER BY avg_usd DESC
    """).df().to_dict(orient="records")

    # 5. Top 5 Skill Combinations
    skills_df = con.execute("""
        SELECT 
            CASE 
                WHEN has_ai = 1 AND has_python = 1 THEN 'Python + AI/LLM'
                WHEN has_cloud = 1 AND has_data = 1 THEN 'Cloud + Data Eng'
                WHEN has_web3 = 1 THEN 'Web3 & Solidity'
                WHEN has_python = 1 AND has_cloud = 1 THEN 'Python + Cloud/DevOps'
                ELSE 'Fullstack & Outros'
            END as skill_cluster,
            round(avg(salary_usd), 0) as avg_usd,
            count(*) as count
        FROM salaries
        GROUP BY skill_cluster
        ORDER BY avg_usd DESC
    """).df().to_dict(orient="records")
    
    # Overview stats
    summary = con.execute("""
        SELECT 
            count(*) as total_jobs,
            round(avg(salary_usd), 0) as avg_salary_usd,
            round(avg(salary_brl), 0) as avg_salary_brl,
            round(avg(ai_impact_index), 1) as avg_ai_impact,
            round(sum(CASE WHEN work_model LIKE '%Remote%' THEN 1 ELSE 0 END) * 100.0 / count(*), 1) as remote_pct
        FROM salaries
    """).df().to_dict(orient="records")[0]

    con.close()
    return JSONResponse({
        "summary": summary,
        "roles": role_df,
        "seniorities": sen_df,
        "regions": region_df,
        "english": eng_df,
        "skills": skills_df,
        "clusters": cluster_stats,
        "metrics": metrics
    })

@app.post("/api/predict")
async def predict_salary(req: PredictRequest):
    # Prepare input DF
    ai_idx = 85.0 if req.has_ai else (65.0 if req.has_data or req.has_cloud else 45.0)
    input_dict = {
        "role": [req.role],
        "seniority": [req.seniority],
        "region": [req.region],
        "work_model": [req.work_model],
        "english_level": [req.english_level],
        "company_size": [req.company_size],
        "years_experience": [req.years_experience],
        "ai_impact_index": [ai_idx],
        "has_python": [req.has_python],
        "has_ai": [req.has_ai],
        "has_cloud": [req.has_cloud],
        "has_data": [req.has_data],
        "has_web3": [req.has_web3],
    }
    input_df = pd.DataFrame(input_dict)
    predicted_usd = float(pipeline.predict(input_df)[0])
    predicted_brl = round(predicted_usd * 5.80, 2)
    monthly_usd = round(predicted_usd / 12, 2)
    monthly_brl = round(predicted_brl / 12, 2)

    # Determine K-Means cluster
    cluster_in = np.array([[
        req.years_experience, ai_idx, req.has_python, req.has_ai, req.has_cloud, req.has_data, req.has_web3, predicted_usd
    ]])
    cluster_scaled = cluster_scaler.transform(cluster_in)
    cluster_id = int(kmeans.predict(cluster_scaled)[0])
    cluster_info = cluster_stats[cluster_id]

    # Percentile benchmarks
    con = duckdb.connect(DB_PATH)
    pct_rank = con.execute(f"SELECT round(count(*) * 100.0 / (SELECT count(*) FROM salaries), 1) FROM salaries WHERE salary_usd <= {predicted_usd}").fetchone()[0]
    con.close()

    return JSONResponse({
        "salary_usd_annual": round(predicted_usd, 2),
        "salary_brl_annual": round(predicted_brl, 2),
        "salary_usd_monthly": monthly_usd,
        "salary_brl_monthly": monthly_brl,
        "percentile_rank": pct_rank,
        "cluster_id": cluster_id,
        "cluster_name": cluster_info["name"],
        "cluster_desc": cluster_info["desc"],
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Tech Career & Salary Intelligence na porta 8002...")
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=False)
