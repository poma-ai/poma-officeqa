"""Every POMA gold label must point at a chunkset ROW that holds the answer needles.

`bench/evaluation_index.get_golden_indices` resolves `local_id` as a row position
(offset + local_id) into the document's chunksets.json. The `chunkset_index` field
stored on each row is NOT that position (2,585 of 2,746 rows differ in
treasury_bulletin_1982_03), and on 2026-05-15 an audit wrote chunkset_index values
into two labels, silently moving them off the evidence for four months. This test
resolves each label the way the benchmark does and checks the needles are there.
"""
import json, re, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence_requirements.json"
POMA_DIR = ROOT / "data" / "poma"

def _norm(s: str) -> str:
    return re.sub(r"[,.$\s]", "", str(s))

def _rows(doc: str) -> list[dict]:
    z = zipfile.ZipFile(POMA_DIR / f"{doc}.poma")
    name = next(n for n in z.namelist() if n.endswith("chunksets.json"))
    return json.loads(z.read(name))

def test_every_poma_gold_row_holds_its_needles():
    """The UNION of a question's labelled rows must hold every needle.

    Labels may list a context row beside the needle row (UID0009, UID0042,
    UID0066 do), so the check is per question, not per row. Dot/comma
    insensitive: ground truth writes 12.57 where the 1982 bulletin prints 12,57.
    """
    ev = json.load(open(EVIDENCE))["UIDs"]
    cache: dict[str, list[dict]] = {}
    checked = 0
    failures = []
    for uid, q in ev.items():
        needles = [str(v) for v in (q.get("ground_truth") or {}).get("values") or []]
        if not needles:
            continue
        for d in q.get("documents", []):
            entries = d.get("found_in", {}).get("poma", [])
            if not entries:
                continue
            union = ""
            labelled = []
            for ent in entries:
                doc, pos = ent["doc"], int(ent["local_id"])
                rows = cache.setdefault(doc, _rows(doc))
                if pos >= len(rows):
                    failures.append(f"{uid}: local_id {pos} beyond {doc} ({len(rows)} rows)")
                    continue
                union += _norm(rows[pos].get("contents") or rows[pos].get("to_embed") or "")
                labelled.append(f"{pos}(chunkset_index={rows[pos].get('chunkset_index')})")
            missing = [n for n in needles if _norm(n) not in union]
            checked += 1
            if missing:
                failures.append(f"{uid}: {doc} rows {labelled} missing {missing}")
    assert checked > 0
    assert not failures, "\n".join(failures)


def test_the_two_restored_labels_and_their_impostors():
    """Regression for a8f5c6f: the row at the *chunkset_index* is not the evidence."""
    rows = _rows("treasury_bulletin_1982_03")
    for right, wrong, needles in ((500, 611, ["80579", "58860"]), (933, 1045, ["12,57", "12,62"])):
        assert rows[right]["chunkset_index"] == wrong, "fixture drifted: the confusion this pins no longer exists"
        good = _norm(rows[right]["contents"]); bad = _norm(rows[wrong]["contents"])
        assert all(_norm(n) in good for n in needles)
        assert not any(_norm(n) in bad for n in needles)
