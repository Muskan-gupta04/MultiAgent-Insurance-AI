import py_compile
from pathlib import Path

FILES = [
    Path("src/setup.py"),
    Path("src/data_pipeline.py"),
    Path("src/agents.py"),
    Path("src/workflow.py"),
    Path("run_project.py"),
]

for file_path in FILES:
    py_compile.compile(str(file_path), doraise=True)
    print(f"OK: {file_path}")

print("SMOKE TEST PASSED")
