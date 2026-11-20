import sys
import subprocess
import os

os.chdir("F:\\AI & ML\\Gen AI\\MediTrust\\backend")
result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_doctors.py", "tests/test_chat.py", "tests/test_public_doctors.py", "tests/test_feedback.py", "tests/test_feedback_legacy.py", "-v", "--tb=short"], capture_output=True, text=True, timeout=300)
print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)