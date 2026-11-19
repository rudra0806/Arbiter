@echo off
REM Runs the whole benchmark on the local model, then on the API model.
REM Cached replies are skipped, so only new or failed tasks are called.
echo ===== LOCAL MODEL =====
python -m benchmark.run_eval --model local
echo.
echo ===== API MODEL =====
python -m benchmark.run_eval --model api
echo.
echo Done. Check the results folder.
pause
