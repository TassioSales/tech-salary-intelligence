@echo off
chcp 65001 > nul
title Tech Career & Global Remote Salary Intelligence - 2026 Live
color 0A

echo ==============================================================================
echo   🌐 TECH CAREER & GLOBAL REMOTE SALARY INTELLIGENCE (2026 BENCHMARK)
echo   DuckDB Analytics + Random Forest ML + Modern Interactive Dashboard
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Verificando dependências Python...
py -3.13 -m pip install -q -r requirements.txt

echo.
echo [2/3] Verificando banco DuckDB e modelos de Machine Learning...
if not exist "data\tech_salaries.duckdb" (
    echo Gerando base colunar com 15.000 vagas no DuckDB...
    py -3.13 data\generate_dataset.py
)
if not exist "ml\models.joblib" (
    echo Treinando modelo preditivo Random Forest e Clusters K-Means...
    py -3.13 ml\train_models.py
)

echo.
echo [3/3] Iniciando servidor do Dashboard em http://127.0.0.1:8002...
start http://127.0.0.1:8002

echo.
echo Servidor ativo! Pressione Ctrl+C para encerrar.
py -3.13 -m uvicorn app.main:app --host 127.0.0.1 --port 8002
pause
