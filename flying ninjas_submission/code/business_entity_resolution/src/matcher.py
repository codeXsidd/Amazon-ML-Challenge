import numpy as np
import lightgbm as lgb
from .config import LGBM_PARAMS, THRESHOLD_CANDIDATES
from .features import FEATURE_NAMES
from .evaluation import compute_entity_f05


class EntityMatcher:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.feature_names = FEATURE_NAMES

    def train(self, X, y):
        self.model = lgb.LGBMClassifier(**LGBM_PARAMS)
        self.model.fit(X, y, feature_name=self.feature_names)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, val_scores, val_s1_ids, val_candidate_ids, val_gt):
        best_f05 = -1
        best_threshold = 0.5

        for threshold in THRESHOLD_CANDIDATES:
            predictions = {}
            for i, (s1_id, cand_id, score) in enumerate(
                zip(val_s1_ids, val_candidate_ids, val_scores)
            ):
                if score >= threshold:
                    if s1_id not in predictions:
                        predictions[s1_id] = set()
                    predictions[s1_id].add(cand_id)

            for s1_id in val_gt:
                if s1_id not in predictions:
                    predictions[s1_id] = set()

            scores = []
            for s1_id, true_matches in val_gt.items():
                pred = predictions.get(s1_id, set())
                scores.append(compute_entity_f05(pred, true_matches))

            macro_f05 = np.mean(scores)
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold, best_f05

    def feature_importance(self):
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, imp),
            key=lambda x: -x[1]
        ))
