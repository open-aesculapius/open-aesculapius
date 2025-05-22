#!/bin/bash

cd ..

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade build

cp -f build/pyproject.toml pyproject.toml

python -m build

if [ -f "pyproject.toml" ]; then
    rm -f pyproject.toml
fi

cd build

echo
echo "Build complete."
