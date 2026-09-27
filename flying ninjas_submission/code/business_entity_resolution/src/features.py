import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def overlap_coeff(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    minimum = min(len(set_a), len(set_b))
    return inter / minimum if minimum > 0 else 0.0


def safe_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def safe_partial_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def safe_token_sort_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def safe_token_set_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def length_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    return min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0.0


def first_token_match(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0
    sorted_a = sorted(tokens_a)
    sorted_b = sorted(tokens_b)
    return 1.0 if sorted_a[0] == sorted_b[0] else 0.0


def compute_pair_features(s1_row, s2s3_row, blocking_methods=None):
    features = {}

    n1 = s1_row['norm_name']
    n2 = s2s3_row['norm_name']
    nf1 = s1_row['norm_name_full']
    nf2 = s2s3_row['norm_name_full']
    nt1 = s1_row['name_tokens']
    nt2 = s2s3_row['name_tokens']
    sn1 = s1_row['sorted_name']
    sn2 = s2s3_row['sorted_name']
    cn1 = s1_row['compact_name']
    cn2 = s2s3_row['compact_name']

    features['name_exact'] = 1.0 if n1 and n2 and n1 == n2 else 0.0
    features['name_sorted_exact'] = 1.0 if sn1 and sn2 and sn1 == sn2 else 0.0
    features['name_ratio'] = safe_ratio(n1, n2)
    features['name_partial_ratio'] = safe_partial_ratio(n1, n2)
    features['name_token_sort'] = safe_token_sort_ratio(n1, n2)
    features['name_token_set'] = safe_token_set_ratio(n1, n2)
    features['name_jaccard'] = jaccard(nt1, nt2)
    features['name_overlap'] = overlap_coeff(nt1, nt2)
    features['name_len_ratio'] = length_ratio(n1, n2)
    features['name_full_ratio'] = safe_ratio(nf1, nf2)
    features['name_first_token'] = first_token_match(nt1, nt2)
    features['compact_name_ratio'] = safe_ratio(cn1, cn2)

    a1 = s1_row['norm_addr']
    a2 = s2s3_row['norm_addr']
    at1 = s1_row['addr_tokens']
    at2 = s2s3_row['addr_tokens']
    an1 = s1_row['addr_numbers']
    an2 = s2s3_row['addr_numbers']

    features['addr_exact'] = 1.0 if a1 and a2 and a1 == a2 else 0.0
    features['addr_ratio'] = safe_ratio(a1, a2)
    features['addr_token_sort'] = safe_token_sort_ratio(a1, a2)
    features['addr_jaccard'] = jaccard(at1, at2)
    features['addr_overlap'] = overlap_coeff(at1, at2)
    features['addr_len_ratio'] = length_ratio(a1, a2)
    features['addr_num_jaccard'] = jaccard(an1, an2)
    features['addr_num_overlap'] = overlap_coeff(an1, an2)

    addr_missing_1 = 1.0 if not a1 else 0.0
    addr_missing_2 = 1.0 if not a2 else 0.0
    features['addr_missing_any'] = max(addr_missing_1, addr_missing_2)
    features['addr_missing_both'] = addr_missing_1 * addr_missing_2

    c1 = s1_row['country_lower']
    c2 = s2s3_row['country_lower']
    features['country_match'] = 1.0 if c1 and c2 and c1 == c2 else 0.0

    features['name_x_addr'] = features['name_token_sort'] * features['addr_token_sort']
    features['name_x_addr_jaccard'] = features['name_jaccard'] * features['addr_jaccard']

    source = s2s3_row['entity_id'][:2]
    features['is_s2'] = 1.0 if source == 'S2' else 0.0

    if blocking_methods is not None:
        features['n_blocking_methods'] = float(len(blocking_methods))
        features['has_exact_block'] = 1.0 if 'exact_name' in blocking_methods else 0.0
        features['has_sorted_block'] = 1.0 if 'sorted_name' in blocking_methods else 0.0
        features['has_token_block'] = 1.0 if 'token_overlap' in blocking_methods else 0.0
        features['has_prefix_block'] = 1.0 if 'prefix' in blocking_methods else 0.0
    else:
        features['n_blocking_methods'] = 0.0
        features['has_exact_block'] = 0.0
        features['has_sorted_block'] = 0.0
        features['has_token_block'] = 0.0
        features['has_prefix_block'] = 0.0

    return features


FEATURE_NAMES = list(compute_pair_features(
    {'norm_name': '', 'norm_name_full': '', 'name_tokens': set(),
     'sorted_name': '', 'compact_name': '',
     'norm_addr': '', 'addr_tokens': set(), 'addr_numbers': set(),
     'country_lower': '', 'entity_id': 'S1-0'},
    {'norm_name': '', 'norm_name_full': '', 'name_tokens': set(),
     'sorted_name': '', 'compact_name': '',
     'norm_addr': '', 'addr_tokens': set(), 'addr_numbers': set(),
     'country_lower': '', 'entity_id': 'S2-0'},
    set()
).keys())
