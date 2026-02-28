# decision.py
import numpy as np

class DecisionAgent:
    def __init__(self, approve_threshold=0.35, manual_review_threshold=0.60):
        """
        approve_threshold: prob_default below which we auto-approve
        manual_review_threshold: prob_default below which we send to manual review;
                                 above it -> Reject
        """
        self.approve_threshold = float(approve_threshold)
        self.manual_review_threshold = float(manual_review_threshold)

    def make_decision(self, prob, features=None):
        """
        Basic decision policy:
          - prob < approve_threshold -> APPROVE
          - approve_threshold <= prob < manual_review_threshold -> MANUAL_REVIEW
          - prob >= manual_review_threshold -> REJECT

        Optional simple feature-based override:
          - If features suggest very strong credit (e.g., high limit & excellent on-time payment feature),
            and prob is not extreme, allow APPROVE.
        features: pandas Series or dict (optional)
        """
        p = float(prob)

        # Feature-based positive override (explainable, conservative)
        try:
            if features is not None:
                # supports both pandas Series and dict
                get = lambda k, default=np.nan: (features.get(k, default) if isinstance(features, dict) else features.get(k, default))

                limit_bal = get("LIMIT_BAL", get("credit_limit", np.nan))
                on_time_rate = get("ON_TIME_PAYMENT_RATE", get("on_time_rate", np.nan))  # expected 0..1

                # conservative override: if applicant has very high limit AND very high on-time rate,
                # they can be approved even if prob is in manual range (but not if prob is already extreme)
                if (not np.isnan(limit_bal) and limit_bal >= 100000) and (not np.isnan(on_time_rate) and on_time_rate >= 0.9):
                    if p < min(self.manual_review_threshold, 0.75):
                        return "APPROVE"
        except Exception:
            # if any feature issue, fallback to standard logic
            pass

        # Standard decision logic
        if p < self.approve_threshold:
            return "APPROVE"
        elif p < self.manual_review_threshold:
            return "MANUAL_REVIEW"
        else:
            return "REJECT"
