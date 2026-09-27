import pandas as pd
from .normalization import (
    normalize_name, get_name_tokens, get_sorted_name,
    normalize_address, get_address_tokens, get_address_numbers,
    get_compact_name, normalize_name_keep_suffix,
)


def load_source(path):
    df = pd.read_csv(path, sep='\t', dtype=str)
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    return df


def preprocess_source(df):
    df = df.copy()
    df['norm_name'] = df['business_name'].apply(normalize_name)
    df['norm_name_full'] = df['business_name'].apply(normalize_name_keep_suffix)
    df['sorted_name'] = df['norm_name'].apply(get_sorted_name)
    df['compact_name'] = df['norm_name'].apply(get_compact_name)
    df['name_tokens'] = df['norm_name'].apply(get_name_tokens)
    df['norm_addr'] = df['business_address'].apply(normalize_address)
    df['addr_tokens'] = df['norm_addr'].apply(get_address_tokens)
    df['addr_numbers'] = df['norm_addr'].apply(get_address_numbers)
    df['country_lower'] = df['country'].str.lower().str.strip()
    return df


def load_ground_truth(path):
    gt = pd.read_csv(path, sep='\t', dtype=str)
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        mids = row['matched_entity_ids']
        if mids:
            gt_dict[s1_id] = set(mids.split(','))
        else:
            gt_dict[s1_id] = set()
    return gt_dict
