import sys
import subprocess
import os

os.chdir("F:\\AI & ML\\Gen AI\\MediTrust\\backend")
result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_public_doctors.py::test_list_all_doctors_public", "-v"], capture_output=True, text=True, timeout=60)
print(result.stdout)
print(result.stderr)
print("Return code:", result.returncode)