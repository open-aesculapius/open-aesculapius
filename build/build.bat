@echo off
setlocal

cd ..

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Upgrade essential build tools
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade build

REM Build the project
copy /y "build\pyproject.toml" "pyproject.toml"
python -m build

if exist "pyproject.toml" (
    del "pyproject.toml" > nul 2>&1
)

cd build

echo.
echo Build complete.
pause