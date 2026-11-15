import faiss
import json
import pickle
import numpy as np
import os
import shutil
from datetime import datetime

# Load current metadata
with open('data/vector_store/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

with open('data/vector_store/id_map.pkl', 'rb') as f:
    id_map = pickle.load(f)

index = faiss.read_index('data/vector_store/index.faiss')

print('Current state:')
print('  FAISS vectors:', index.ntotal)
print('  Metadata chunks:', len(metadata))
print('  ID map entries:', len(id_map))

# Rebuild from metadata (should be 5 entries)
vectors = []
new_id_map = {}
for new_idx, (chunk_id, faiss_idx) in enumerate(sorted(id_map.items(), key=lambda x: x[1])):
    vec = index.reconstruct(faiss_idx)
    vectors.append(vec)
    new_id_map[chunk_id] = new_idx

vectors_arr = np.array(vectors, dtype=np.float32)
print('Rebuilding with', len(vectors), 'vectors')

# Backup current
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
for fname in ['index.faiss', 'metadata.json', 'id_map.pkl']:
    shutil.copy2('data/vector_store/' + fname, 'data/vector_store/' + fname + '.pre_cleanup_' + ts)

# Write new index
new_index = faiss.IndexFlatIP(384)
new_index.add(vectors_arr)
faiss.write_index(new_index, 'data/vector_store/index.faiss')

# Write new id_map
with open('data/vector_store/id_map.pkl', 'wb') as f:
    pickle.dump(new_id_map, f)

# Verify
new_idx = faiss.read_index('data/vector_store/index.faiss')
with open('data/vector_store/id_map.pkl', 'rb') as f:
    new_map = pickle.load(f)

print('After rebuild:')
print('  FAISS vectors:', new_idx.ntotal)
print('  ID map entries:', len(new_map))
print('SUCCESS' if new_idx.ntotal == 5 else 'FAIL')