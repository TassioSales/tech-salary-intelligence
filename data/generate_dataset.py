"""
Data Generator & Ingestion Pipeline for Tech Career & Global Remote Salary Intelligence.
Uses DuckDB for ultra-fast columnar analytical queries.
"""
import os
import duckdb
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "tech_salaries.duckdb")

ROLES = [
    ("AI / LLM Engineer", 1.35, ["Python", "PyTorch", "LangChain", "RAG", "FastAPI", "VectorDB", "HuggingFace"]),
    ("Machine Learning Engineer", 1.25, ["Python", "TensorFlow", "Scikit-Learn", "Docker", "MLflow", "AWS SageMaker"]),
    ("Data Engineer", 1.20, ["Python", "DuckDB", "SQL", "Snowflake", "dbt", "Apache Spark", "Airflow"]),
    ("Cloud & DevOps Architect", 1.22, ["AWS", "Kubernetes", "Terraform", "Docker", "CI/CD", "Linux", "Golang"]),
    ("Backend Engineer (Python/Go)", 1.10, ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Go", "Kafka"]),
    ("Fullstack Engineer (TypeScript)", 1.05, ["TypeScript", "React", "Next.js", "Node.js", "Tailwind", "PostgreSQL"]),
    ("Web3 & Smart Contract Engineer", 1.28, ["Solidity", "EVM", "Web3.py", "Rust", "Foundry", "Ethereum"]),
    ("Mobile Engineer (Flutter/Swift)", 1.00, ["Flutter", "Dart", "Swift", "iOS", "Android", "Firebase"]),
]

SENIORITIES = [
    ("Junior (0-2y)", 0.60, 0.20),
    ("Pleno / Mid (3-5y)", 0.95, 0.40),
    ("Senior (6-9y)", 1.40, 0.25),
    ("Lead / Staff (10y+)", 1.85, 0.12),
    ("Principal / Director", 2.30, 0.03),
]

REGIONS = [
    ("United States", 1.50, 0.35, "EUA"),
    ("Western Europe", 1.15, 0.25, "Europa"),
    ("Latin America / Brasil (Remoto US$)", 0.85, 0.25, "Brasil"),
    ("Canada", 1.20, 0.10, "América do Norte"),
    ("Asia-Pacific", 0.95, 0.05, "APAC"),
]

WORK_MODELS = [
    ("100% Remote Global", 1.10, 0.50),
    ("Remote National", 0.90, 0.25),
    ("Hybrid", 0.95, 0.15),
    ("Onsite", 1.00, 0.10),
]

ENGLISH_LEVELS = [
    ("Fluent / Native", 1.25, 0.45),
    ("Advanced", 1.10, 0.35),
    ("Intermediate", 0.85, 0.15),
    ("Basic", 0.65, 0.05),
]

COMPANY_SIZES = [
    ("Enterprise / Big Tech", 1.30, 0.25),
    ("Scaleup (Series B/C)", 1.15, 0.35),
    ("Startup (Seed/Series A)", 0.95, 0.25),
    ("Digital Agency / Consultancy", 0.85, 0.15),
]

def generate_benchmark_data(num_samples=15000):
    np.random.seed(42)
    records = []
    
    role_choices = [r[0] for r in ROLES]
    role_weights = [0.18, 0.14, 0.16, 0.14, 0.15, 0.12, 0.06, 0.05]
    role_dict = {r[0]: (r[1], r[2]) for r in ROLES}
    
    sen_names = [s[0] for s in SENIORITIES]
    sen_weights = [s[2] for s in SENIORITIES]
    sen_multipliers = {s[0]: s[1] for s in SENIORITIES}
    
    reg_names = [r[0] for r in REGIONS]
    reg_weights = [r[2] for r in REGIONS]
    reg_multipliers = {r[0]: (r[1], r[3]) for r in REGIONS}
    
    wm_names = [w[0] for w in WORK_MODELS]
    wm_weights = [w[2] for w in WORK_MODELS]
    wm_multipliers = {w[0]: w[1] for w in WORK_MODELS}
    
    eng_names = [e[0] for e in ENGLISH_LEVELS]
    eng_weights = [e[2] for e in ENGLISH_LEVELS]
    eng_multipliers = {e[0]: e[1] for e in ENGLISH_LEVELS}
    
    comp_names = [c[0] for c in COMPANY_SIZES]
    comp_weights = [c[2] for c in COMPANY_SIZES]
    comp_multipliers = {c[0]: c[1] for c in COMPANY_SIZES}

    roles = np.random.choice(role_choices, size=num_samples, p=role_weights)
    seniorities = np.random.choice(sen_names, size=num_samples, p=sen_weights)
    regions = np.random.choice(reg_names, size=num_samples, p=reg_weights)
    work_models = np.random.choice(wm_names, size=num_samples, p=wm_weights)
    english = np.random.choice(eng_names, size=num_samples, p=eng_weights)
    company_sizes = np.random.choice(comp_names, size=num_samples, p=comp_weights)
    years_exp_base = {"Junior (0-2y)": 1.5, "Pleno / Mid (3-5y)": 4.0, "Senior (6-9y)": 7.5, "Lead / Staff (10y+)": 11.0, "Principal / Director": 15.0}

    base_salary = 95000.0  # US$ mid baseline

    for i in range(num_samples):
        role = roles[i]
        sen = seniorities[i]
        reg = regions[i]
        wm = work_models[i]
        eng = english[i]
        comp = company_sizes[i]

        r_mult, skills_pool = role_dict[role]
        s_mult = sen_multipliers[sen]
        reg_mult, geo_group = reg_multipliers[reg]
        wm_mult = wm_multipliers[wm]
        eng_mult = eng_multipliers[eng]
        c_mult = comp_multipliers[comp]

        # Years of experience with realistic jitter
        exp_mean = years_exp_base[sen]
        exp_years = max(0.5, round(np.random.normal(exp_mean, 1.0), 1))

        # Select 3-5 key skills from role pool
        k_skills = np.random.choice(skills_pool, size=np.random.randint(3, len(skills_pool) + 1), replace=False)
        skills_str = ", ".join(k_skills)

        # AI Exposure Index (0-100%): higher for AI/LLM, lower for legacy mobile/ops
        if "AI" in role or "Machine Learning" in role:
            ai_impact = np.random.uniform(75, 98)
            ai_premium = 1.15
        elif "Data" in role or "Cloud" in role or "Web3" in role:
            ai_impact = np.random.uniform(50, 80)
            ai_premium = 1.05
        else:
            ai_impact = np.random.uniform(30, 65)
            ai_premium = 0.98

        noise = np.random.normal(1.0, 0.08)
        salary_usd = round(base_salary * r_mult * s_mult * reg_mult * wm_mult * eng_mult * c_mult * ai_premium * noise, -2)
        salary_brl = round(salary_usd * 5.80, -2)

        records.append({
            "id": f"JOB-{10000 + i}",
            "role": role,
            "seniority": sen,
            "region": reg,
            "geo_group": geo_group,
            "work_model": wm,
            "english_level": eng,
            "company_size": comp,
            "years_experience": exp_years,
            "skills": skills_str,
            "skills_count": len(k_skills),
            "ai_impact_index": round(ai_impact, 1),
            "salary_usd": float(salary_usd),
            "salary_brl": float(salary_brl),
            "year": 2026,
            "has_python": int("Python" in skills_str),
            "has_ai": int(any(k in skills_str for k in ["PyTorch", "LangChain", "RAG", "TensorFlow", "VectorDB"])),
            "has_cloud": int(any(k in skills_str for k in ["AWS", "Kubernetes", "Docker", "Terraform", "CI/CD"])),
            "has_data": int(any(k in skills_str for k in ["DuckDB", "SQL", "Snowflake", "dbt", "Spark"])),
            "has_web3": int("Solidity" in skills_str or "Web3.py" in skills_str or "Ethereum" in skills_str),
        })

    df = pd.DataFrame(records)
    
    # Ingest into DuckDB
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS salaries")
    con.execute("CREATE TABLE salaries AS SELECT * FROM df")
    count = con.execute("SELECT count(*) FROM salaries").fetchone()[0]
    avg_sal = con.execute("SELECT avg(salary_usd) FROM salaries").fetchone()[0]
    con.close()
    
    print(f"✅ DuckDB criado com sucesso em: {DB_PATH}")
    print(f"📊 Total de Vagas Ingeridas: {count:,} registros")
    print(f"💰 Salário Médio Global: US$ {avg_sal:,.2f} / ano")
    return df

if __name__ == "__main__":
    generate_benchmark_data()
