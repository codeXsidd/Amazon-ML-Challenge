import numpy as np


def compute_entity_f05(predicted_set, true_set):
    if not true_set and not predicted_set:
        return 1.0
    if not true_set and predicted_set:
        return 0.0
    if true_set and not predicted_set:
        return 0.0

    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def evaluate_predictions(predictions, ground_truth):
    scores = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_total = 0

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())

        f05 = compute_entity_f05(pred_matches, true_matches)
        scores.append(f05)

        tp = len(pred_matches & true_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        if not true_matches:
            singleton_total += 1
            if not pred_matches:
                singleton_correct += 1

    macro_f05 = np.mean(scores) if scores else 0.0
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    return {
        'macro_f05': macro_f05,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'n_entities': len(ground_truth),
        'singleton_total': singleton_total,
        'singleton_correct': singleton_correct,
        'singleton_accuracy': singleton_correct / singleton_total if singleton_total > 0 else 1.0,
    }


def evaluate_candidates(candidate_dict, ground_truth):
    total_recall_hits = 0
    total_true_matches = 0
    total_candidates = 0
    n_entities = 0

    for s1_id, true_matches in ground_truth.items():
        candidates = candidate_dict.get(s1_id, set())
        total_candidates += len(candidates)
        n_entities += 1

        if true_matches:
            hits = len(true_matches & candidates)
            total_recall_hits += hits
            total_true_matches += len(true_matches)

    candidate_recall = total_recall_hits / total_true_matches if total_true_matches > 0 else 0.0
    avg_candidates = total_candidates / n_entities if n_entities > 0 else 0.0

    return {
        'candidate_recall': candidate_recall,
        'avg_candidates': avg_candidates,
        'total_candidates': total_candidates,
        'n_entities': n_entities,
    }
