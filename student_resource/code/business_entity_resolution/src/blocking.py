from collections import defaultdict
from .config import TOKEN_FREQ_CAP, MIN_SHARED_TOKENS, MAX_CANDIDATES_PER_S1, NAME_PREFIX_LEN


def build_inverted_index(entities):
    token_index = defaultdict(set)
    for eid, tokens in entities:
        for t in tokens:
            token_index[t].add(eid)
    return token_index


def build_name_index(entities):
    name_index = defaultdict(set)
    for eid, norm_name in entities:
        if norm_name:
            name_index[norm_name].add(eid)
    return name_index


def build_sorted_name_index(entities):
    idx = defaultdict(set)
    for eid, sorted_name in entities:
        if sorted_name:
            idx[sorted_name].add(eid)
    return idx


def build_prefix_index(entities, prefix_len=NAME_PREFIX_LEN):
    idx = defaultdict(set)
    for eid, compact_name in entities:
        if compact_name and len(compact_name) >= prefix_len:
            prefix = compact_name[:prefix_len]
            idx[prefix].add(eid)
    return idx


def build_addr_number_index(entities):
    idx = defaultdict(set)
    for eid, numbers in entities:
        for n in numbers:
            if len(n) >= 3:
                idx[n].add(eid)
    return idx


def generate_candidates_for_entity(
    s1_id, s1_norm_name, s1_sorted_name, s1_compact_name,
    s1_name_tokens, s1_addr_numbers,
    name_index, sorted_name_index, prefix_index,
    token_index, token_freq, addr_num_index,
    max_candidates=MAX_CANDIDATES_PER_S1,
):
    candidates = set()
    method_counts = defaultdict(set)

    if s1_norm_name and s1_norm_name in name_index:
        exact = name_index[s1_norm_name]
        candidates.update(exact)
        for c in exact:
            method_counts[c].add('exact_name')

    if s1_sorted_name and s1_sorted_name in sorted_name_index:
        sn = sorted_name_index[s1_sorted_name]
        candidates.update(sn)
        for c in sn:
            method_counts[c].add('sorted_name')

    if s1_compact_name and len(s1_compact_name) >= NAME_PREFIX_LEN:
        prefix = s1_compact_name[:NAME_PREFIX_LEN]
        if prefix in prefix_index:
            pf = prefix_index[prefix]
            candidates.update(pf)
            for c in pf:
                method_counts[c].add('prefix')

    useful_tokens = [
        t for t in s1_name_tokens
        if t in token_freq and token_freq[t] < TOKEN_FREQ_CAP and len(t) > 1
    ]

    if useful_tokens:
        token_candidates = defaultdict(int)
        for t in useful_tokens:
            for eid in token_index.get(t, []):
                token_candidates[eid] += 1

        min_overlap = max(MIN_SHARED_TOKENS, min(2, len(useful_tokens)))
        for eid, count in token_candidates.items():
            if count >= min_overlap:
                candidates.add(eid)
                method_counts[eid].add('token_overlap')

    if s1_addr_numbers:
        for num in s1_addr_numbers:
            if len(num) >= 4 and num in addr_num_index:
                addr_cands = addr_num_index[num]
                candidates.update(addr_cands)
                for c in addr_cands:
                    method_counts[c].add('addr_number')

    candidates.discard(s1_id)

    if len(candidates) > max_candidates:
        scored = []
        for c in candidates:
            score = len(method_counts.get(c, set()))
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        candidates = {c for _, c in scored[:max_candidates]}

    return candidates, {c: method_counts.get(c, set()) for c in candidates}


class BlockingEngine:
    def __init__(self, s2s3_df):
        self.s2s3_df = s2s3_df
        self._build_indexes()

    def _build_indexes(self):
        df = self.s2s3_df
        ents_tokens = list(zip(df['entity_id'], df['name_tokens']))
        ents_name = list(zip(df['entity_id'], df['norm_name']))
        ents_sorted = list(zip(df['entity_id'], df['sorted_name']))
        ents_compact = list(zip(df['entity_id'], df['compact_name']))
        ents_addr_nums = list(zip(df['entity_id'], df['addr_numbers']))

        self.name_index = build_name_index(ents_name)
        self.sorted_name_index = build_sorted_name_index(ents_sorted)
        self.prefix_index = build_prefix_index(ents_compact)
        self.token_index = build_inverted_index(ents_tokens)
        self.addr_num_index = build_addr_number_index(ents_addr_nums)

        self.token_freq = {t: len(ids) for t, ids in self.token_index.items()}

    def generate_candidates(self, s1_row):
        return generate_candidates_for_entity(
            s1_id=s1_row['entity_id'],
            s1_norm_name=s1_row['norm_name'],
            s1_sorted_name=s1_row['sorted_name'],
            s1_compact_name=s1_row['compact_name'],
            s1_name_tokens=s1_row['name_tokens'],
            s1_addr_numbers=s1_row['addr_numbers'],
            name_index=self.name_index,
            sorted_name_index=self.sorted_name_index,
            prefix_index=self.prefix_index,
            token_index=self.token_index,
            token_freq=self.token_freq,
            addr_num_index=self.addr_num_index,
        )
