'''parse_times.py -- pull the per-station homogenization wall-time from a homo run log into
times_<key>.csv (<tag>,<seconds>) in the matching homo_<key>/ folder, so emit_timo_out.py can print it.
    python parse_times.py --key rm --log ~/claude_tmp/hrm.log
The homo scripts print lines like:  [iea_r0247 ] EA=...  [58.8s]
'''
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = {"rm": "homo_rm", "jax": "homo_jax", "fenics": "homo_fenics"}
PAT = re.compile(r"\[\s*(iea_r\d+)\s*\].*\[([\d.]+)\s*s\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=list(FOLDER))
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    rows = []
    for ln in open(os.path.expanduser(a.log)):
        m = PAT.search(ln)
        if m:
            rows.append((m.group(1), m.group(2)))
    out = os.path.join(HERE, FOLDER[a.key], "times_%s.csv" % a.key)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for tag, t in rows:
            f.write("%s,%s\n" % (tag, t))
    print("wrote %s (%d stations)" % (out, len(rows)))


if __name__ == "__main__":
    main()
