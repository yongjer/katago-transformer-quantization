"""Turn selfplay .sgfs (one SGF per line) into KataGo analysis-engine queries for 19x19 games.

One query per game, analyzing every `--every`-th turn, so a run touches a spread of opening,
middlegame and endgame positions.
"""
import argparse, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--sgfs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--visits", type=int, default=800)
ap.add_argument("--every", type=int, default=12)
ap.add_argument("--max-positions", type=int, default=200)
args = ap.parse_args()

COLS = "ABCDEFGHJKLMNOPQRST"
def loc(xy, size):
    if xy == "" or xy == "tt": return "pass"
    x, y = ord(xy[0]) - 97, ord(xy[1]) - 97
    return f"{COLS[x]}{size - y}"

queries, npos = [], 0
for gi, line in enumerate(open(args.sgfs)):
    line = line.strip()
    if not line: continue
    sz = int(re.search(r"SZ\[(\d+)", line).group(1))
    if sz != 19: continue
    km = float(re.search(r"KM\[([-\d.]+)", line).group(1))
    root = line.split(";B[")[0].split(";W[")[0]
    init = []
    for color, block in re.findall(r"A([BW])((?:\[[a-s]{2}\])+)", root):
        init += [[color, loc(p, sz)] for p in re.findall(r"\[([a-s]{2})\]", block)]
    moves = [[c, loc(p, sz)] for c, p in re.findall(r";([BW])\[([a-s]{0,2})\]", line)]
    turns = list(range(args.every // 2, len(moves), args.every))
    if not turns: continue
    turns = turns[: args.max_positions - npos]
    q = {"id": f"g{gi}", "moves": moves, "initialStones": init, "rules": "tromp-taylor", "komi": km,
         "boardXSize": 19, "boardYSize": 19, "maxVisits": args.visits, "analyzeTurns": turns,
         "includePolicy": False, "overrideSettings": {"rootSymmetryPruning": False}}
    queries.append(q); npos += len(turns)
    if npos >= args.max_positions: break
with open(args.out, "w") as f:
    for q in queries: f.write(json.dumps(q) + "\n")
print(f"{len(queries)} games, {npos} positions -> {args.out}")
