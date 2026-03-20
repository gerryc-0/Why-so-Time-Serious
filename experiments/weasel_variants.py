"""
WEASEL 2.0 variant classifiers.

Each class inherits from WEASEL_V2 and overrides only the minimal
code needed to inject the proposed improvement. This keeps the
experimental comparison fair: only one thing changes at a time.

To use these, you must have `weasel-classifier` installed:
    pip install weasel-classifier

The base class lives at:
    weasel.classification.dictionary_based._weasel_v2.WEASEL_V2
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from weasel.classification.dictionary_based import WEASEL_V2


# ============================================================================
# Variant 1: TF-IDF weighting on the word-count feature matrix
# ============================================================================

class WEASEL_V2_TFIDF(WEASEL_V2):
    """WEASEL 2.0 + TF-IDF weighting (use_idf=True, sublinear_tf=False)."""

    def _fit(self, X, y):
        # Run the standard WEASEL 2.0 fit to build SFA transformers
        # and produce the concatenated word-count matrix.
        # We intercept just before the classifier is trained.

        # Call the parent _fit which builds self.SFA_transformers,
        # concatenates features, and trains self.clf.
        # We need to replicate the parent logic with our insertion point.
        super()._fit(X, y)

        # After parent fit, the classifier is already trained on raw counts.
        # We need to retrain it on TF-IDF transformed features.
        # So we re-transform the training data and refit.
        all_words = self._transform_words(X)

        self._tfidf = TfidfTransformer(use_idf=True, sublinear_tf=False)
        all_words_tfidf = self._tfidf.fit_transform(all_words)

        # Retrain the classifier on the TF-IDF features.
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.clf.fit(all_words_tfidf, y)
        return self

    def _predict(self, X):
        bag = self._transform_words(X)
        bag = self._tfidf.transform(bag)
        return self.clf.predict(bag)

    def _predict_proba(self, X):
        bag = self._transform_words(X)
        bag = self._tfidf.transform(bag)
        return self.clf.predict_proba(bag)


# ============================================================================
# Variant 2: Sublinear TF only (no IDF component)
# ============================================================================

class WEASEL_V2_SublinearTF(WEASEL_V2):
    """WEASEL 2.0 + sublinear TF scaling: 1 + log(count)."""

    def _fit(self, X, y):
        super()._fit(X, y)

        all_words = self._transform_words(X)

        self._tfidf = TfidfTransformer(use_idf=False, sublinear_tf=True)
        all_words_tf = self._tfidf.fit_transform(all_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.clf.fit(all_words_tf, y)
        return self

    def _predict(self, X):
        bag = self._transform_words(X)
        bag = self._tfidf.transform(bag)
        return self.clf.predict(bag)

    def _predict_proba(self, X):
        bag = self._transform_words(X)
        bag = self._tfidf.transform(bag)
        return self.clf.predict_proba(bag)


# ============================================================================
# Variant 3: Replace Ridge with SGDClassifier
# ============================================================================

class WEASEL_V2_SGD(WEASEL_V2):
    """WEASEL 2.0 with SGDClassifier instead of RidgeClassifierCV."""

    def _fit(self, X, y):
        super()._fit(X, y)

        # Retrain with SGD instead of Ridge
        all_words = self._transform_words(X)

        self.clf = SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.clf.fit(all_words, y)
        return self

    # _predict and _predict_proba use the same interface, no override needed.


# ============================================================================
# Variant 4: Disable first-order differences (halves feature space)
# ============================================================================

class WEASEL_V2_NoDiff(WEASEL_V2):
    """
    WEASEL 2.0 without applying dilation to first-order differences.

    The parent class doubles the feature space by running SFA on both
    the raw dilated series and its first-order differences. This variant
    disables the differencing, halving memory and features.

    NOTE: The exact mechanism depends on how the parent implements
    first_difference. You may need to inspect the parent code and
    override the relevant parameter. Common options:

        - Set self.use_first_differences = False (if the parent exposes it)
        - Override the internal _parallel_fit to skip the diff branch

    The code below shows the pattern; adjust the attribute name to
    match the actual parent implementation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Try to disable first differences via known attribute names.
        # If the parent uses a different name, update this.
        if hasattr(self, "use_first_differences"):
            self.use_first_differences = False
        if hasattr(self, "first_difference"):
            self.first_difference = False
