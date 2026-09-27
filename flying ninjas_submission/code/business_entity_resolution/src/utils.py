import os
import pandas as pd


def validate_output(matching_path, candidate_path, test_s1_path):
    """Quick internal validation of output files."""
    errors = []

    s1_ids = set()
    with open(test_s1_path, encoding='utf-8') as f:
        next(f)
        for line in f:
            eid = line.split('\t', 1)[0].strip()
            if eid:
                s1_ids.add(eid)

    with open(matching_path, encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'source1_entity_id\tmatched_entity_ids':
            errors.append(f'Bad matching header: {header}')

        seen = set()
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            s1 = parts[0]
            if s1 in seen:
                errors.append(f'Duplicate S1: {s1}')
            seen.add(s1)

            if len(parts) > 1 and parts[1].strip():
                mids = parts[1].split(',')
                for mid in mids:
                    if not mid.startswith(('S2-', 'S3-')):
                        errors.append(f'Invalid match ID: {mid}')
                if len(mids) != len(set(mids)):
                    errors.append(f'Duplicate match IDs for {s1}')

        missing = s1_ids - seen
        extra = seen - s1_ids
        if missing:
            errors.append(f'Missing {len(missing)} S1 entities')
        if extra:
            errors.append(f'Extra {len(extra)} S1 entities')

    return errors
