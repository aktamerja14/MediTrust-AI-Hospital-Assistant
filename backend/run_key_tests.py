import sys
import subprocess
import os

os.chdir("F:\\AI & ML\\Gen AI\\MediTrust\\backend")
result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_doctors.py", "tests/test_public_doctors.py", "-v", "--tb=short"], capture_output=True, text=True, timeout=120)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)