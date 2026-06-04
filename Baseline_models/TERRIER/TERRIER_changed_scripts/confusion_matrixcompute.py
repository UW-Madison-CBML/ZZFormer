from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class MapRule:
    src: str
    dst: str
    level: str  # "superfamily" or "order"


def parse_map_rules(map_str: str) -> list[MapRule]:
    """
    MAP format: comma-separated key=value pairs.
    - If key starts with "/", apply to superfamily token (last path segment). Example: /I-Jockey=/I
    - Else apply to order token (first path segment). Example: Helitron=RC
    """
    rules: list[MapRule] = []
    map_str = (map_str or "").strip()
    if not map_str:
        return rules

    for raw in map_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Bad map rule (missing '='): {raw}")

        left, right = raw.split("=", 1)
        left = left.strip()
        right = right.strip()

        if left.startswith("/"):
            rules.append(MapRule(src=left[1:], dst=right.lstrip("/"), level="superfamily"))
        else:
            rules.append(MapRule(src=left, dst=right, level="order"))

    return rules


def tokenize(label: str) -> tuple[str, str]:
    """
    Returns (order, superfamily) for a label like 'LINE/I' or 'LTR/Gypsy'.
    If no '/', superfamily is same as order.
    """
    label = (label or "").strip()
    if not label:
        return ("Unknown", "Unknown")
    if "/" in label:
        parts = [p for p in label.split("/") if p != ""]
        if not parts:
            return ("Unknown", "Unknown")
        order = parts[0]
        superfamily = parts[-1]
        return (order, superfamily)
    return (label, label)


def apply_mapping(label: str, rules: list[MapRule]) -> str:
    """
    Apply mapping to a full hierarchical label.
    - order-level rules rewrite the first token (order)
    - superfamily-level rules rewrite the last token (superfamily)
    """
    order, superfamily = tokenize(label)

    for r in rules:
        if r.level == "order" and order == r.src:
            order = r.dst
        elif r.level == "superfamily" and superfamily == r.src:
            superfamily = r.dst

    # If original label had hierarchy, keep hierarchy; otherwise keep single-token
    if "/" in (label or ""):
        return f"{order}/{superfamily}"
    else:
        # if mapping changed superfamily independently, prefer order-only output
        return order


def pick_level(label: str, superfamily: bool) -> str:
    order, sf = tokenize(label)
    return sf if superfamily else order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="Terrier predictions CSV (already has prediction column).")
    ap.add_argument("--map", default="", dest="map_rules", help="Comma-separated map rules.")
    ap.add_argument("--superfamily", action="store_true", help="Compute confusion matrix at superfamily level.")
    ap.add_argument("--ignore", default="Unknown", help="Ignore this true label (after mapping/level selection).")
    ap.add_argument("--output", default=None, type=Path, help="Optional output CSV path for confusion matrix.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Required columns
    for col in ["prediction", "original_classification"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {args.csv}")

    rules = parse_map_rules(args.map_rules)

    # Map full labels first, then select level (order vs superfamily token)
    true_full = df["original_classification"].fillna("Unknown").astype(str).map(lambda s: apply_mapping(s, rules))
    pred_full = df["prediction"].fillna("Unknown").astype(str).map(lambda s: apply_mapping(s, rules))

    true_lbl = true_full.map(lambda s: pick_level(s, superfamily=args.superfamily))
    pred_lbl = pred_full.map(lambda s: pick_level(s, superfamily=args.superfamily))

    # Ignore rows with ignored true label
    mask = true_lbl != args.ignore
    true_lbl = true_lbl[mask]
    pred_lbl = pred_lbl[mask]

    # Confusion matrix (counts)
    cm = pd.crosstab(true_lbl, pred_lbl, rownames=["true"], colnames=["pred"], dropna=False)

    # Make it square-ish (include all labels in both axes)
    labels = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    print(cm)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cm.to_csv(args.output)
        print(f"\nWrote confusion matrix to: {args.output}")


if __name__ == "__main__":
    main()