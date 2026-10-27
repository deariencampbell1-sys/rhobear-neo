"""trusted_reviewers — which outside reviewers count as green 2 (Wave 3 API).

The customer's merge authority (board `merge-authority` v7, `trusted`) is
"Apollo ACCEPT on the head, plus the repo's own merge authority — architect
approval or a trusted reviewer's green. Either one counts."  Which OUTSIDE
reviewers count is the customer's choice (owner Q5: "CodeAnt counts as green
for us, and which outside reviewers count is a customer's choice they can
turn off later") — that choice lives in this table:

    trusted_reviewers(install_id, repo, source, enabled, updated_at)

`repo='*'` is the install-wide wildcard; a repo-exact row overrides it per
source.  The schema is created inline (ensure_schema, no migrations dir) and
seeded ONCE with the owner install's CodeAnt row — ON CONFLICT DO NOTHING, so
a customer's explicit disabled row is never flipped back on.

list_sources / set_source / delete_source are install-scoped for Wave 3
(who is on my list, add one, remove one).  Tests never touch a real
database: they replace _connect with a recording fake.
"""
from __future__ import annotations

import re

import psycopg

from .config import DEFAULT_OWNER_INSTALL_ID

TABLE = "trusted_reviewers"

#: The one seeded source: the owner install's CodeAnt (W1-07 step 2).
SEED_SOURCE = "CodeAnt"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    install_id text NOT NULL,
    repo text NOT NULL DEFAULT '*',
    source text NOT NULL,
    enabled boolean NOT NULL DEFAULT true,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (install_id, repo, source)
)
"""

#: Idempotent seed: never re-enables a row the customer disabled (defect:
#: flipping an explicit False would be changing a customer's evidence).
SEED_SQL = f"""
INSERT INTO {TABLE} (install_id, repo, source, enabled, updated_at)
VALUES (%s, '*', %s, true, now())
ON CONFLICT (install_id, repo, source) DO NOTHING
"""

UPSERT_SQL = f"""
INSERT INTO {TABLE} (install_id, repo, source, enabled, updated_at)
VALUES (%s, %s, %s, %s, now())
ON CONFLICT (install_id, repo, source) DO UPDATE
SET enabled = EXCLUDED.enabled, updated_at = now()
"""

LIST_SQL = (f"SELECT install_id, repo, source, enabled, updated_at FROM {TABLE} "
            "WHERE install_id = %s")

DELETE_SQL = f"DELETE FROM {TABLE} WHERE install_id = %s AND repo = %s AND source = %s"

_COLUMNS = ("install_id", "repo", "source", "enabled", "updated_at")


def _connect(dsn: str):
    """One connection per call (the NeoState/Jobs pattern). Tests replace this."""
    return psycopg.connect(dsn)


def ensure_schema(dsn: str, *, owner_install_id: str | None = DEFAULT_OWNER_INSTALL_ID) -> None:
    """Create the table if missing and seed the owner install's CodeAnt row.

    Idempotent and self-healing: the seed is ON CONFLICT DO NOTHING, so this
    is safe to call on every read/write and never overrides a customer's
    explicit choice."""
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            if owner_install_id is not None:
                cur.execute(SEED_SQL, (owner_install_id, SEED_SOURCE))
        conn.commit()


def _rows(dsn: str, install_id: str) -> list[dict]:
    ensure_schema(dsn, owner_install_id=None)
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(LIST_SQL, (install_id,))
            return [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]


def list_sources(dsn: str, install_id: str, repo: str = "*") -> list[dict]:
    """This install's rows for one scope (repo='*' = the install-wide rows).

    Install-scoped: rows of another install are never visible here."""
    return [r for r in _rows(dsn, install_id) if str(r.get("repo") or "*") == repo]


def set_source(dsn: str, install_id: str, repo: str, source: str,
               enabled: bool = True) -> None:
    """Turn one source on or off for one scope (upsert)."""
    ensure_schema(dsn, owner_install_id=None)
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(UPSERT_SQL, (install_id, repo, source, bool(enabled)))
        conn.commit()


def delete_source(dsn: str, install_id: str, repo: str, source: str) -> None:
    """Remove one source from one scope."""
    ensure_schema(dsn, owner_install_id=None)
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(DELETE_SQL, (install_id, repo, source))
        conn.commit()


def enabled_sources(dsn: str, install_id: str, repo: str) -> tuple[str, ...]:
    """The sources that count as green 2 for this install + repo.

    A repo-exact row overrides the install-wide '*' row per source (off or
    on); a source with no row counts for nothing.  Sorted for determinism."""
    best: dict[str, bool] = {}
    for r in _rows(dsn, install_id):
        src = str(r.get("source") or "")
        if not src:
            continue
        scope = str(r.get("repo") or "*")
        if scope != "*" and scope != repo:
            continue                          # another repo's row says nothing here
        on = bool(r.get("enabled"))
        if scope == "*":
            best.setdefault(src, on)          # the wildcard is the fallback
        else:
            best[src] = on                    # a repo row overrides it
    return tuple(sorted(s for s, on in best.items() if on))


# --- matching a source to what GitHub names it ------------------------------
# "CodeAnt" ↔ "code-ant" / "CodeAnt[bot]"; "Greptile" ↔ "greptile[bot]".
# An unknown account never matches (a PR author cannot name themselves green).

def norm_name(name: str) -> str:
    s = str(name or "").lower()
    s = s.replace("[bot]", " ").replace("-bot", " ").replace("_bot", " ")
    return re.sub(r"[^a-z0-9]", "", s)


#: Explicit alias rows (W1-07-r2 defect 1): when GitHub names a source
#: differently enough that norm_name cannot reconcile it, that source's
#: customer row gets an EXPLICIT alias here — never a heuristic prefix.  A
#: prefix match would let a GitHub account named `ap`, `apol`, `apollox` or
#: `apollobot2` spoof the Apollo ACCEPT that authorises a merge.
ALIASES: dict[str, tuple[str, ...]] = {
    # CodeAnt's customer-facing name differs from its GitHub App slug/login.
    "CodeAnt": ("codeant-ai",),
}


def source_matches(source: str, name: str) -> bool:
    """True when a status context / check name / review author belongs to the
    trusted source.  Normalised EQUALITY only (plus the source's explicit
    ALIASES row) — never a prefix in either direction ("ap" must not match
    "apollo", "apollox" must not match "apollo"), never a substring match
    ("CodeAnt" does not match "codecov").  The legitimate tolerance is all in
    norm_name: [bot] / -bot / _bot suffixes and separators ("CodeAnt" ↔
    "code-ant" ↔ "CodeAnt[bot]")."""
    a, b = norm_name(source), norm_name(name)
    if not a or not b:
        return False                      # an empty author/name matches nothing
    if a == b:
        return True
    return b in {norm_name(alias) for alias in ALIASES.get(str(source or ""), ())}
