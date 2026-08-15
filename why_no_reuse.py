"""Why is each conversation not being reused?  python why_no_reuse.py [db_path]"""
import sqlite3, sys, time

path = sys.argv[1] if len(sys.argv) > 1 else "second_brain.db"
conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
PLACEHOLDERS = ("", "new conversation", "new conversation (main)")

rows = conn.execute("""
    SELECT c.*, (SELECT COUNT(*) FROM conversation_messages m
                  WHERE m.conversation_id = c.id
                    AND (m.role IS NULL OR m.role <> 'system')) AS real_rows
      FROM conversations c ORDER BY c.updated_at DESC LIMIT 25""").fetchall()

now = time.time()
print(f"{path}: {len(rows)} most recent conversations\n")
for r in rows:
    why = []
    if (r["kind"] or "user") != "user":          why.append(f"kind={r['kind']!r}")
    if r["real_rows"]:                           why.append(f"{r['real_rows']} real message rows")
    if (r["category"] or "") != "":              why.append(f"category={r['category']!r}")
    if (r["title"] or "").strip().lower() not in PLACEHOLDERS:
                                                 why.append(f"title={r['title']!r}")
    quiet = now - (r["updated_at"] or r["created_at"] or 0)
    if quiet < 30:                               why.append(f"touched {quiet:.0f}s ago (quiet window)")
    verdict = "REUSABLE" if not why else "no: " + ", ".join(why)
    print(f"  #{r['id']:<5} user={str(r['user_id']):<5} {verdict}")

n = sum(1 for r in rows if not (
    (r["kind"] or "user") != "user" or r["real_rows"] or (r["category"] or "")
    or (r["title"] or "").strip().lower() not in PLACEHOLDERS
    or now - (r["updated_at"] or r["created_at"] or 0) < 30))
print(f"\n{n} conversation(s) pass every on-disk test.")
print("If that is >0 and /new still makes a new row, the only clause left is\n"
      "live sessions: a frontend leaving one behind per conversation.")
