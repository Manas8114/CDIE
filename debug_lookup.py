import sys
import os
from pathlib import Path
os.environ["CDIE_DATA_DIR"] = str(Path().cwd() / "data")
sys.path.insert(0, str(Path(r"c:\Users\msgok\OneDrive\Desktop\Project\hackathon\Rename")))
from cdie.api.lookup import SafetyMapLookup  # type: ignore
from cdie.api.intent_parser import classify_query  # type: ignore

q = "What happens if SIM box fraud attempts increase by 30%?"
c = classify_query(q)
print("Classification:", c)

lkp = SafetyMapLookup()
scen, is_ext = lkp.find_best_scenario(c["source"], c["target"], c["magnitude"])
print("Scenario ID:", scen["id"] if scen else None)

import sqlite3
with sqlite3.connect(lkp.db_path) as conn:
    curs = conn.cursor()
    key = f"{c['source']}__{c['target']}__increase_{int(c['magnitude'])}"
    print("Expected Key:", key)
    curs.execute("SELECT id FROM scenarios WHERE id=?", (key,))
    print("DB lookup direct:", curs.fetchone())
