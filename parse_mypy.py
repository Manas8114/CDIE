import subprocess
r = subprocess.run(["mypy", "."], capture_output=True, text=True, encoding="utf-8", errors="replace")
errors = [line for line in r.stdout.splitlines() if "error:" in line]
with open("mypy_parsed.txt", "w", encoding="utf-8") as f:
    for e in errors:
        f.write(e + "\n")
    f.write(f"\n--- Total: {len(errors)} errors ---\n")
print(f"Wrote {len(errors)} errors to mypy_parsed.txt")
