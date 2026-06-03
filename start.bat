@echo off
echo ============================================
echo   Predictor de Viabilidad Financiera
echo   Sistema de IA para Prediccion Crediticia
echo ============================================
echo.
echo [1/3] Verificando dependencias...
pip install -r backend\requirements.txt > nul 2>&1
echo [OK] Dependencias instaladas
echo.
echo [2/3] Iniciando servidor backend...
start "FinPredict API" python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
echo [OK] API iniciada en http://localhost:8000
echo.
echo [3/3] Abriendo frontend...
start http://localhost:8000
start "" "frontend\index.html"
echo.
echo ============================================
echo   Sistema listo para usar
echo   API: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ============================================
pause
