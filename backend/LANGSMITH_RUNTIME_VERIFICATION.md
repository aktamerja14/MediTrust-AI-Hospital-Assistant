# LangSmith Runtime Verification Procedure

This document describes the safe manual verification procedure for LangSmith integration using the existing LangSmith project, approved synthetic dataset, and valid API credentials.

## Prerequisites

- Existing LangSmith project configured
- Approved synthetic dataset only (no real patient information)
- Valid LangSmith API credentials (`LANGSMITH_API_KEY` environment variable)
- No real patient information in any test data

## Verification Steps

### 1. Dataset Verification
```bash
# List datasets in workspace
python -m app.cli.evaluation list --api-key $LANGSMITH_API_KEY

# Verify dataset exists
python -m app.cli.evaluation validate-dataset --file data/evaluation_dataset.jsonl --output validation_report.json
```

**Expected**: Dataset `meditrust-rag-evaluation-v1` exists with synthetic evaluation cases only.

### 2. Dataset Upload with PHI Sanitization
```bash
# Dry-run to preview what would be uploaded
python -m app.cli.evaluation dry-run --file data/evaluation_dataset.jsonl --output dryrun_report.json

# Upload with sanitization (default, fail-closed)
python -m app.cli.evaluation upload-dataset --file data/evaluation_dataset.jsonl --api-key $LANGSMITH_API_KEY

# Verify no sensitive data in uploaded examples
python -m app.cli.evaluation get-dataset-examples --dataset meditrust-rag-evaluation-v1 --api-key $LANGSMITH_API_KEY --limit 5
```

**Expected**: 
- Dry-run shows only synthetic cases
- Upload succeeds with PHI sanitization applied
- No SSN, MRN, patient names, or medical diagnoses in uploaded data

### 3. Native Experiment Execution
```bash
# Run native LangSmith experiment
python -m app.cli.evaluation native-experiment \
  --name "verification-exp-$(date +%s)" \
  --dataset meditrust-rag-evaluation-v1 \
  --output experiment_results.json \
  --max-examples 5 \
  --max-concurrent 2
```

**Expected**: 
- Experiment created successfully in LangSmith
- Experiment ID and URL captured in output
- Results file contains experiment summary

### 4. LangSmith UI Verification
Navigate to the LangSmith project and verify:

1. **Dataset**: `meditrust-rag-evaluation-v1` exists with expected example count
2. **Experiment**: New experiment appears with name matching `--name` parameter
3. **Runs**: Evaluation runs visible under the experiment
4. **Evaluator Feedback**: Feedback visible for each run (correctness, faithfulness, relevance, medical_safety)
5. **Feedback IDs**: Feedback records have IDs accessible via SDK

### 5. Feedback ID Retrieval Test
```python
# In Python REPL or script
from app.ai.evaluation.langsmith_native import NativeLangSmithExperimentRunner, NativeExperimentConfig
from app.ai.evaluation.langsmith_utils import create_langsmith_client

client = create_langsmith_client()
# List feedback for experiment runs
feedback = list(client.list_feedback(run_ids=["run-id-1", "run-id-2"]))
for f in feedback:
    print(f"Feedback ID: {f.id}, Run: {f.run_id}, Key: {f.key}, Score: {f.score}")
```

**Expected**: Feedback IDs retrieved and mapped to run IDs, example IDs, evaluator names, and scores.

### 6. Privacy Verification
```bash
# Check uploaded data for sensitive content
python -c "
from app.ai.evaluation.langsmith_utils import get_dataset_examples
examples = get_dataset_examples('meditrust-rag-evaluation-v1', api_key='\$LANGSMITH_API_KEY', limit=10)
for ex in examples:
    print(f'Question: {ex[\"inputs\"][\"question\"][:100]}')
    print(f'Expected: {ex[\"outputs\"][\"expected_answer\"][:100]}')
    print(f'Context: {ex[\"outputs\"][\"reference_context\"][:100]}')
    print('---')
"
```

**Expected**: No SSN, MRN, patient names, real medical diagnoses, or medication dosages in any field.

### 7. Error Handling Verification

#### 7.1 Authentication Failure
```bash
# Test with invalid API key
python -m app.cli.evaluation list --api-key "invalid-key"
```

**Expected**: Clear error message identifying authentication issue (401/403), privacy controls remain active.

#### 7.2 Sanitization Failure
```bash
# Create test file with PHI
echo '{"id": "test_phi", "question": "Patient John Doe SSN 123-45-6789", "expected_answer": "Cannot process", "reference_context": "PHI protection", "category": "edge_cases", "language": "en", "tags": ["phi"], "needs_human_verification": true}' > test_phi.jsonl

# Attempt upload - should fail
python -m app.cli.evaluation upload-dataset --file test_phi.jsonl --api-key $LANGSMITH_API_KEY
```

**Expected**: Upload blocked with `SanitizationUploadError`, no data sent to LangSmith.

## Verification Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Dataset exists in LangSmith | ☐ | |
| Experiment created successfully | ☐ | |
| Experiment URL and ID captured | ☐ | |
| Evaluation runs visible | ☐ | |
| Evaluator feedback visible | ☐ | |
| Feedback IDs retrievable | ☐ | |
| No sensitive data in dataset | ☐ | |
| No sensitive data in experiment runs | ☐ | |
| Auth failure handled safely | ☐ | |
| Sanitization failure blocks upload | ☐ | |

## Troubleshooting

### 401/403 Errors
- Verify `LANGSMITH_API_KEY` is set and valid
- Check workspace/project permissions
- Ensure API key has read/write access to datasets and experiments
- **Do not disable privacy controls** to bypass authentication

### Missing Feedback IDs
- Feedback IDs may not be available immediately after experiment completion
- Wait a few seconds and retry
- Verify evaluators are configured to produce feedback (all default evaluators do)

### Sanitization False Positives
- The sanitization patterns are conservative (fail-closed)
- If legitimate content is blocked, review patterns in `langsmith_utils.py`
- Use `--no-sanitize` flag only for testing with known-safe data

## Notes

- This verification uses **only synthetic data** - no real patient information
- All PHI protection is application-level - not a substitute for HIPAA compliance
- The `--no-sanitize` flag exists for development only; never use in production
- Feedback ID retrieval depends on LangSmith SDK v0.15.0 API compatibility