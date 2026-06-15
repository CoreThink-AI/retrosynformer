uv pip install sympy==1.13.1
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
uv pip install --upgrade -e ../rdchiral/
uv pip install --upgrade numpy rdkit pandas dask dill requests scipy scikit-learn matplotlib seaborn markupsafe optuna mypy-extensions partd dask huggingface-hub filelock cycler contourpy
