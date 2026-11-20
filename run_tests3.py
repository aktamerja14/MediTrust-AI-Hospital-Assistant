import sys
import os

backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)
os.chdir(backend_path)

import pytest

# Run the tests
exit_code = pytest.main([
    "tests/test_feedback.py",
    "tests/test_admin_dashboard.py",
    "tests/test_documents.py",
    "-v",
    "--tb=short"
])

sys.exit(exit_code)