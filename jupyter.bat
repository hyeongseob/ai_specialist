@echo off
rem 복습용 Jupyter Lab 실행 (최초 실행 시 환경 자동 설치)
cd /d "%~dp0"
uv sync
uv run jupyter lab
