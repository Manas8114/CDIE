import json
with open('ruff_errors.json', 'r', encoding='utf-16') as f:
    data = json.load(f)
for m in data:
    print(f"{m['filename']}:{m['location']['row']}:{m['location']['column']} - {m['code']} - {m['message']}")
