"""
Phase 1 data pipeline for fill-the-gap.

Usage:
  python data/prepare.py                      # real data from the Hub
  python data/prepare.py --near-dedup         # also MinHash near-duplicate removal (slower)
  python data/prepare.py --local-jsonl x.jsonl   # offline testing with a local file
"""
import argparse, hashlib, json, os, random, re
import numpy as np

SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<MASK>", "<PRE>", "<SUF>", "<MID>"]
OUT = os.path.dirname(os.path.abspath(__file__))


def load_files(args):
    if args.local_jsonl:
        with open(args.local_jsonl) as f:
            return [json.loads(l) for l in f]
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    return [{"repo_name": r["repo_name"], "path": r["path"], "content": r["content"]} for r in ds]


def norm_hash(text):
    # Collapse all whitespace so files that differ only in spacing count as duplicates.
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def near_dedup(files, threshold=0.85, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    lsh, keep = MinHashLSH(threshold=threshold, num_perm=num_perm), []
    for i, f in enumerate(files):
        words = re.findall(r"\w+", f["content"])
        shingles = {" ".join(words[j:j + 5]) for j in range(max(1, len(words) - 4))}
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode())
        if not lsh.query(m):          # nothing similar seen yet -> keep it
            lsh.insert(str(i), m)
            keep.append(f)
    return keep


def repo_bucket(repo):
    # Deterministic 0-99 bucket from the repo name: same repo always lands in the same split.
    return int(hashlib.md5(repo.encode()).hexdigest(), 16) % 100


def train_tokenizer(texts, vocab_size, path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,          # ids 0..5, in this order
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


def to_windows(tok, texts, seq_len, max_tokens=None):
    eos = tok.token_to_id("<EOS>")
    ids, n = [], 0
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids + [eos])
        n += len(enc.ids) + 1
        if max_tokens and n >= max_tokens:
            break
    n_win = len(ids) // seq_len
    return np.array(ids[: n_win * seq_len], dtype=np.uint16).reshape(n_win, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-jsonl")
    p.add_argument("--near-dedup", action="store_true")
    p.add_argument("--vocab", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-tokens", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)

    files = load_files(args)
    stats = {"raw_files": len(files)}

    files = [f for f in files if f["path"].endswith(".py") and len(f["content"].encode()) < 50_000]
    stats["after_filter"] = len(files)

    seen, uniq = set(), []
    for f in files:
        h = norm_hash(f["content"])
        if h not in seen:
            seen.add(h)
            uniq.append(f)
    files = uniq
    stats["after_exact_dedup"] = len(files)

    if args.near_dedup:
        files = near_dedup(files)
        stats["after_near_dedup"] = len(files)

    split = {"train": [], "val": [], "test": []}
    for f in files:
        b = repo_bucket(f["repo_name"])
        split["test" if b < 5 else "val" if b < 10 else "train"].append(f)
    for k in split:
        random.shuffle(split[k])     # fixed seed -> same order every run
        stats[f"{k}_files"] = len(split[k])
        stats[f"{k}_repos"] = len({f["repo_name"] for f in split[k]})

    # Sanity check: no repo appears in two splits.
    repos = {k: {f["repo_name"] for f in v} for k, v in split.items()}
    assert not (repos["train"] & repos["val"]) and not (repos["train"] & repos["test"]) \
        and not (repos["val"] & repos["test"]), "repo leak between splits!"

    train_texts = [f["content"] for f in split["train"]]
    tok = train_tokenizer(train_texts, args.vocab, os.path.join(OUT, "tokenizer", "tokenizer.json"))

    # Train pool: first ~5M tokens = budget, next ~5M = fresh-data control.
    pool = to_windows(tok, train_texts, args.seq_len, max_tokens=2 * args.train_tokens)
    n_budget = args.train_tokens // args.seq_len
    train, control = pool[:n_budget], pool[n_budget:]
    val = to_windows(tok, [f["content"] for f in split["val"]], args.seq_len)

    np.save(os.path.join(OUT, "train.npy"), train)
    np.save(os.path.join(OUT, "control.npy"), control)
    np.save(os.path.join(OUT, "val.npy"), val)
    with open(os.path.join(OUT, "test_files.jsonl"), "w") as fh:
        for f in split["test"]:
            fh.write(json.dumps(f) + "\n")

    stats.update(train_tokens=int(train.size), control_tokens=int(control.size),
                 val_tokens=int(val.size), vocab=tok.get_vocab_size(),
                 special_ids={t: tok.token_to_id(t) for t in SPECIAL_TOKENS})
    with open(os.path.join(OUT, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    if control.size < args.train_tokens * 0.9:
        print("NOTE: not enough data for a full fresh-data control; that's optional.")


if __name__ == "__main__":
    main()