with open(r'F:\AI & ML\Gen AI\MediTrust\backend\app\cli\evaluation.py', 'r') as f:
    lines = f.readlines()

# Fix line 895 (0-indexed: 894)
lines[894] = '    upload_dedup_parser.add_argument("--update-existing", action="store_true",\n'

# Fix line 1088 (0-indexed: 1087)
lines[1087] = '    upload_dedup_parser.add_argument("--update-existing", action="store_true",\n'

with open(r'F:\AI & ML\Gen AI\MediTrust\backend\app\cli\evaluation.py', 'w') as f:
    f.writelines(lines)

print('Fixed indentation')