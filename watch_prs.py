"""Watch new real PRs on the active personal repos and log how the auto-gate handles each.
Durable (nohup) VPS-side watcher: detects PRs opened after start, tracks review verdict +
Neo action, flags gaps (reviewed but Neo silent). Appends concise lines to watch.log.
"""
import json, os, subprocess, time

REPOS = "rhobear-hub-web rhobear-plans-web rhobear-plans rhobear-designs rhobear-designs-web rhobear-sales-chat rhobear-app rhobear-scribe rhobear-strive rhobear-academy-gate rhobear-cloud-web rhobear-cloud-workbench rhobear-reviews rhobear-workbench-router neo-dogfood rhobear rhobear-capturd rhobear-companion rhobear-reviews-app".split()
LOG = "/opt/rhobear/rhobear-neo/watch.log"
DB = subprocess.run("grep -E '^DATABASE_URL=' /opt/rhobear/rhobear-neo/.env | cut -d= -f2-", shell=True, capture_output=True, text=True).stdout.strip()
START = time.time()
END = START + 6 * 3600  # watch up to 6h

def log(m):
    line = time.strftime("%m-%d %H:%M:%S", time.gmtime()) + " " + m
    with open(LOG, "a") as f: f.write(line + "\n")

def gh(args):
    p = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=40)
    return p.stdout.strip()

def neo_state(repo, pr):
    q = f"select phase,verdict from neo_actions where repo like '%{repo}' and pr_number={pr} order by created_at desc limit 1;"
    out = subprocess.run(["psql", DB, "-t", "-c", q], capture_output=True, text=True, timeout=20).stdout.strip()
    return out  # "phase | verdict" or ""

seen = {}   # key -> dict(detected_ts, reviewed, neo_done, gap_flagged)
log(f"=== WATCH START (watching {len(REPOS)} repos for PRs opened after now) ===")
while time.time() < END:
    for r in REPOS:
        try:
            raw = gh(["pr", "list", "-R", f"deariencampbell1-sys/{r}", "--state", "open",
                      "--json", "number,title,author,createdAt,headRefOid,isDraft"])
            prs = json.loads(raw) if raw else []
        except Exception:
            continue
        for pr in prs:
            import datetime
            try:
                cts = datetime.datetime.fromisoformat(pr["createdAt"].replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if cts < START or pr.get("isDraft"):
                continue
            key = f"{r}#{pr['number']}"
            st = seen.setdefault(key, {"detected": time.time(), "reviewed": False, "neo_done": False, "gap": False})
            if st.get("announced") is None:
                log(f"NEW  {key} by {pr['author']['login']} — {pr['title'][:60]}")
                st["announced"] = True
            sha = pr.get("headRefOid", "")
            # reviews status
            try:
                srch = gh(["api", f"repos/deariencampbell1-sys/{r}/commits/{sha}/status"])
                sd = json.loads(srch) if srch else {}
                rev = next((s for s in sd.get("statuses", []) if s.get("context") == "rhobear-reviews"), None)
            except Exception:
                rev = None
            if rev and not st["reviewed"]:
                st["reviewed"] = True; st["reviewed_ts"] = time.time()
                log(f"REV  {key} rhobear-reviews={rev.get('state')}")
            ns = neo_state(r, pr["number"])
            if ns and not st["neo_done"]:
                phase = ns.split("|")[0].strip()
                if phase in ("ready", "merged", "bounced", "escalated", "fix_forward"):
                    st["neo_done"] = True
                    log(f"NEO  {key} -> {ns.strip()}")
            # gap: reviewed >3min ago, no neo action
            if st["reviewed"] and not st["neo_done"] and not st["gap"] and time.time() - st.get("reviewed_ts", time.time()) > 180:
                st["gap"] = True
                log(f"GAP! {key} reviewed but Neo SILENT >3min — gate missed it")
    time.sleep(30)
log("=== WATCH END ===")
