"""
Build the infill test set from held-out repos (data/test_files.jsonl).

Each item = one Python function with a gap:
    prefix  +  [ middle = the hidden part ]  +  suffix
Three gap types:
    line  -- one full line of the function body
    block -- 2 to 5 consecutive body lines
    expr  -- one expression: call arguments, an if/while condition, a return value,
             or the right-hand side of an assignment

Key detail: the split is done in TOKEN space. We tokenize the whole function once and
cut at token boundaries, so every model sees exactly the tokens it would see in normal
code (cutting the text first and tokenizing the pieces separately creates odd tokens
at the seams, e.g. a newline split from its indentation, which unfairly hurts models).

Usage:  python data/build_infill.py            -> data/infill_test.jsonl
"""
import argparse, ast, hashlib, json, os, random, re, textwrap, warnings
from collections import Counter, defaultdict
from tokenizers import Tokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
# Old Python-2-era files trigger harmless "invalid escape sequence" warnings when parsed.
warnings.filterwarnings("ignore", category=SyntaxWarning)
LIMITS = {"line": (2, 40), "block": (6, 80), "expr": (2, 30)}   # middle length in tokens


def functions_in(source):
    """Yield (name, dedented source) for every function that parses on its own."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            src = textwrap.dedent("".join(lines[start - 1:node.end_lineno]))
            if not src.endswith("\n"):
                src += "\n"
            try:
                ast.parse(src)
                yield node.name, src
            except SyntaxError:
                continue


def body_lines(src):
    """Line numbers (0-based) that are real statements: not def/decorator lines, blank
    lines, docstring lines, or lines containing a comment (no model can guess a comment)."""
    tree = ast.parse(src)
    fn = tree.body[0]
    skip = set(range(0, fn.body[0].lineno - 1))              # decorators + signature
    first = fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        skip |= set(range(first.lineno - 1, first.end_lineno))   # docstring
    lines = src.splitlines(keepends=True)
    return [i for i, l in enumerate(lines)
            if i not in skip and l.strip() and "#" not in l], lines


def expr_spans(src):
    """Character spans (start, end) of candidate single-line expressions."""
    tree = ast.parse(src)
    line_start = [0]
    for l in src.splitlines(keepends=True):
        line_start.append(line_start[-1] + len(l))
    off = lambda ln, col: line_start[ln - 1] + col
    spans = []
    for n in ast.walk(tree):
        target = None
        if isinstance(n, ast.Call) and (n.args or n.keywords):
            parts = list(n.args) + [k.value for k in n.keywords]
            a, b = min(parts, key=lambda p: (p.lineno, p.col_offset)), \
                max(parts, key=lambda p: (p.end_lineno, p.end_col_offset))
            target = (a.lineno, a.col_offset, b.end_lineno, b.end_col_offset)
        elif isinstance(n, (ast.If, ast.While)):
            target = (n.test.lineno, n.test.col_offset, n.test.end_lineno, n.test.end_col_offset)
        elif isinstance(n, ast.Return) and n.value is not None:
            v = n.value
            target = (v.lineno, v.col_offset, v.end_lineno, v.end_col_offset)
        elif isinstance(n, ast.Assign):
            v = n.value
            target = (v.lineno, v.col_offset, v.end_lineno, v.end_col_offset)
        if target and target[0] == target[2] and target[0] > 1:   # one line, not the def line
            s, e = off(target[0], target[1]), off(target[2], target[3])
            if 3 <= e - s <= 80:
                spans.append((s, e))
    return spans


def cut(enc, cs, ce):
    """Smallest token range [a, b) covering characters [cs, ce)."""
    a = next((i for i, (s, e) in enumerate(enc.offsets) if e > cs), None)
    b = next((i for i, (s, e) in enumerate(enc.offsets) if s >= ce), len(enc.ids))
    return a, b


def make_item(tok, src, kind, cs, ce):
    enc = tok.encode(src)
    a, b = cut(enc, cs, ce)
    if a is None or a == 0 or b >= len(enc.ids):
        return None
    lo, hi = LIMITS[kind]
    if not lo <= b - a <= hi:
        return None
    p, m, s = enc.ids[:a], enc.ids[a:b], enc.ids[b:]
    item = dict(prefix=tok.decode(p), middle=tok.decode(m), suffix=tok.decode(s),
                prefix_ids=p, middle_ids=m, suffix_ids=s)
    if item["prefix"] + item["middle"] + item["suffix"] != src:   # exact round trip or skip
        return None
    if not item["middle"].strip():
        return None
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-files", default=os.path.join(OUT, "test_files.jsonl"))
    ap.add_argument("--tokenizer", default=os.path.join(OUT, "tokenizer", "tokenizer.json"))
    ap.add_argument("--out", default=os.path.join(OUT, "infill_test.jsonl"))
    ap.add_argument("--per-type", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=240)   # + 3 FIM sentinels must fit in 256
    ap.add_argument("--min-lines", type=int, default=4)
    ap.add_argument("--max-per-repo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)

    files = [json.loads(l) for l in open(args.test_files)]
    rng.shuffle(files)
    items, seen, per_repo, n_funcs = defaultdict(list), set(), Counter(), 0

    for f in files:
        if all(len(items[k]) >= args.per_type for k in LIMITS):
            break
        for name, src in functions_in(f["content"]):
            if per_repo[f["repo_name"]] >= args.max_per_repo:
                break
            h = hashlib.md5(re.sub(r"\s+", " ", src).encode()).hexdigest()
            if h in seen:
                continue
            n_tok = len(tok.encode(src).ids)
            if n_tok > args.max_tokens or src.count("\n") < args.min_lines:
                continue
            seen.add(h)
            n_funcs += 1
            body, lines = body_lines(src)
            starts = [0]
            for l in lines:
                starts.append(starts[-1] + len(l))

            # One gap type per function (rotating), so items are independent functions.
            for kind in rng.sample(list(LIMITS), 3):
                if len(items[kind]) >= args.per_type:
                    continue
                item = None
                if kind == "line" and body:
                    i = rng.choice(body)
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    item = make_item(tok, src, kind, starts[i] + indent, starts[i + 1] - 1)
                elif kind == "block" and len(body) >= 2:
                    k = rng.randint(2, 5)
                    runs = [body[j:j + k] for j in range(len(body) - k + 1)
                            if body[j + k - 1] - body[j] == k - 1]   # k consecutive lines
                    if runs:
                        r = rng.choice(runs)
                        indent = len(lines[r[0]]) - len(lines[r[0]].lstrip())
                        item = make_item(tok, src, kind, starts[r[0]] + indent, starts[r[-1] + 1] - 1)
                elif kind == "expr":
                    spans = expr_spans(src)
                    if spans:
                        item = make_item(tok, src, kind, *rng.choice(spans))
                if item:
                    item.update(id=f"{kind}-{len(items[kind]):04d}", type=kind,
                                repo=f["repo_name"], path=f["path"], func=name)
                    items[kind].append(item)
                    per_repo[f["repo_name"]] += 1
                    break

    allitems = [it for k in LIMITS for it in items[k]]
    with open(args.out, "w") as fh:
        for it in allitems:
            fh.write(json.dumps(it) + "\n")

    print(f"functions considered: {n_funcs} | repos used: {len(per_repo)}")
    for k in LIMITS:
        ml = [len(it["middle_ids"]) for it in items[k]]
        print(f"{k:5s}: {len(ml)} items | middle tokens mean {sum(ml)/max(1,len(ml)):.1f} "
              f"min {min(ml, default=0)} max {max(ml, default=0)}")
    print(f"total: {len(allitems)} -> {args.out}")
    for k in LIMITS:
        if items[k]:
            it = items[k][0]
            print(f"\n===== example [{k}] {it['func']}  (from {it['repo']})")
            print(it["prefix"] + "⟦" + it["middle"] + "⟧" + it["suffix"])


if __name__ == "__main__":
    main()