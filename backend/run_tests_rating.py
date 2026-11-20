import sys
import subprocess
import os

os.chdir("F:\\AI & ML\\Gen AI\\MediTrust\\backend")
result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_feedback_legacy.py", "-v"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print("Return code:", result.returncode)