"""Regex-aware alias loader for label normalization.

A spec is either:
  - a TSV-ish file with one rule per line; separator can be TAB or '='
      LHS<TAB>RHS         or         LHS=RHS
      (lines starting with '#' and blank lines are ignored)
  - an inline string  'a=b,c=d,e/f=g'

Each LHS is treated as a Python regex *anchored to the whole string*.
You can use captures in LHS and refer to them in RHS with \\1, \\2, ...
Plain literal strings still work (they just match themselves).

Application is iterative: rules are tried in order, the first one that
fully matches replaces the label, then we re-apply (up to `max_iter`).
"""
import os, re

def _parse_pairs(spec):
    pairs = []
    if not spec:
        return pairs
    if os.path.isfile(spec):
        with open(spec) as fh:
            for line in fh:
                line = line.split('#', 1)[0].strip()
                if not line: continue
                if '\t' in line:
                    lhs, rhs = line.split('\t', 1)
                elif '=' in line:
                    lhs, rhs = line.split('=', 1)
                else:
                    continue
                pairs.append((lhs.strip(), rhs.strip()))
    else:
        for tok in spec.split(','):
            tok = tok.strip()
            if not tok or '=' not in tok: continue
            lhs, rhs = tok.split('=', 1)
            pairs.append((lhs.strip(), rhs.strip()))
    return pairs


def load_alias_map(spec):
    """Return list of (compiled_regex, replacement) preserving rule order."""
    out = []
    for lhs, rhs in _parse_pairs(spec):
        pat = lhs
        if not pat.startswith('^'): pat = '^' + pat
        if not pat.endswith('$'):   pat = pat + '$'
        try:
            out.append((re.compile(pat), rhs))
        except re.error as e:
            print(f'WARN: skipping invalid regex {lhs!r}: {e}')
    return out


def apply_alias(label, alias_list, max_iter=5):
    """Apply the first matching rule, repeatedly, up to max_iter rounds."""
    if label is None or not alias_list: return label
    for _ in range(max_iter):
        new = label
        for pat, repl in alias_list:
            if pat.match(label):
                new = pat.sub(repl, label)
                break
        if new == label:
            return label
        label = new
    return label