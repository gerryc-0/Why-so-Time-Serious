"""
WEASEL 2.0 variants.
Each class inherits from WEASEL_V2.

Use run_experiments.py to run experiments.
"""

from joblib import Parallel, delayed
from scipy.sparse import hstack

from weasel.classification.dictionary_based._weasel_v2 import _parallel_fit

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from weasel.classification.dictionary_based import WEASEL_V2

class WEASEL_V2_AdaptiveEnsemble(WEASEL_V2):
    """WEASEL 2.0 with new ensemble size selection
    ensemble_size is selected based on series_length and n_classes.
    """

    def __init__(self, ensemble_size=50, max_window=84, min_window=4,
                 norm_options=[False], word_lengths=[7, 8],
                 use_first_differences=[True, False],
                 feature_selection="chi2_top_k", max_feature_count=30_000,
                 random_state=None, n_jobs=4):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y):
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        self.n_classes = len(np.unique(y))
        XX = X.squeeze(1)

        # Original max_window rule
        if self.n_instances < 250:
            self.max_window = 24
        elif self.series_length < 100:
            self.max_window = 44
        else:
            self.max_window = 84

        # New ensemble_size selection
        if self.series_length > 700:
            self.ensemble_size = 75
        elif self.n_classes < 3:
            self.ensemble_size = 25
        else:
            self.ensemble_size = 50

        print(f"[Adaptive n={self.n_instances} len={self.series_length} c={self.n_classes}] "
              f"ensemble_size={self.ensemble_size} max_window={self.max_window}")


        self.max_window = int(min(self.series_length, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window={self.min_window} is bigger than "
                f"max_window={self.max_window}, series length is {self.series_length}"
            )

        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, prefer="threads")(
            delayed(_parallel_fit)(
                i,
                XX,
                y.copy(),
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.lower_bounding,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.remove_repeat_words,
                self.sections,
                self.random_state,
            )
            for i in range(self.ensemble_size)
        )

        sfa_words = []
        for (words, transformer) in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        if type(sfa_words[0]) is np.ndarray:
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            all_words = hstack(sfa_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))
        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[1]
        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self

class WEASEL_V2_Config(WEASEL_V2):
    """WEASEL 2.0 with fixed ensemble_size and max_window"""

    def __init__(self, ensemble_size=50, max_window=84, min_window=4,
                 norm_options=[False], word_lengths=[7, 8],
                 use_first_differences=[True, False],
                 feature_selection="chi2_top_k", max_feature_count=30_000,
                 random_state=None, n_jobs=4):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._cfg_ensemble_size = ensemble_size
        self._cfg_max_window = max_window

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["ensemble_size"] = self._cfg_ensemble_size
        params["max_window"] = self._cfg_max_window
        return params

    def _fit(self, X, y):
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        XX = X.squeeze(1)

        self.ensemble_size = self._cfg_ensemble_size
        self.max_window = int(min(self.series_length, self._cfg_max_window))

        print(f"[Config n={self.n_instances} len={self.series_length}] "
              f"ensemble_size={self.ensemble_size} "
              f"max_window={self.max_window} (requested={self._cfg_max_window})")

        if self.min_window > self.max_window:
            raise ValueError(
                f"min_window={self.min_window} > max_window={self.max_window}, "
                f"series_length={self.series_length}"
            )

        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, prefer="threads")(
            delayed(_parallel_fit)(
                i, XX, y.copy(), self.window_sizes, self.alphabet_sizes,
                self.word_lengths, self.series_length, self.norm_options,
                self.use_first_differences, self.binning_strategies,
                self.variance, self.anova, self.bigrams, self.lower_bounding,
                self.n_jobs, self.max_feature_count, self.ensemble_size,
                self.feature_selection, self.remove_repeat_words,
                self.sections, self.random_state,
            )
            for i in range(self.ensemble_size)
        )

        sfa_words = []
        for (words, transformer) in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        if type(sfa_words[0]) is np.ndarray:
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            all_words = hstack(sfa_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))
        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[1]
        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self


class WEASEL_V2_HalfEnsemble(WEASEL_V2):
    """WEASEL 2.0 with half ensemble size.

    The default rule of thumb sets ensemble_size to 50, 100, or 150 depending
    on dataset size. This variant simply halves each tier: 25, 50, 75.
    """

    def __init__(
            self,
            min_window=4,
            norm_options=[False],
            word_lengths=[7, 8],
            use_first_differences=[True, False],
            feature_selection="chi2_top_k",
            max_feature_count=30_000,
            random_state=None,
            n_jobs=4,
    ):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y):
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        XX = X.squeeze(1)

        # Changed from 50,100,150 to 25,50,75.
        if self.n_instances < 250:
            self.max_window = 24
            self.ensemble_size = 25
        elif self.series_length < 100:
            self.max_window = 44
            self.ensemble_size = 50
        else:
            self.max_window = 84
            self.ensemble_size = 75

        print(f"[HalfEnsemble n={self.n_instances} len={self.series_length} | old={self.ensemble_size * 2} -> new={self.ensemble_size}")

        self.max_window = int(min(self.series_length, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={self.max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )

        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, prefer="threads")(
            delayed(_parallel_fit)(
                i,
                XX,
                y.copy(),
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.lower_bounding,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.remove_repeat_words,
                self.sections,
                self.random_state,
            )
            for i in range(self.ensemble_size)
        )

        sfa_words = []
        for (words, transformer) in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        if type(sfa_words[0]) is np.ndarray:
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            all_words = hstack(sfa_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))
        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[1]
        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self


class WEASEL_V2_FixedEnsemble(WEASEL_V2):
    """WEASEL 2.0 with a fixed ensemble size regardless of dataset characteristics.
    Bypasses the rule of thumb and uses a fixed value for all datasets.
    """

    def __init__(self, r_max=50, min_window=4, norm_options=[False],
                 word_lengths=[7, 8], use_first_differences=[True, False],
                 feature_selection="chi2_top_k", max_feature_count=30_000,
                 random_state=None, n_jobs=4):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.r_max = r_max

    def _fit(self, X, y):
        from joblib import Parallel, delayed
        from scipy.sparse import hstack
        from sklearn.linear_model import RidgeClassifierCV
        from weasel.classification.dictionary_based._weasel_v2 import _parallel_fit

        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        XX = X.squeeze(1)

        # rule-of-thumb block for max_window only -- ensemble_size is fixed
        if self.n_instances < 250:
            self.max_window = 24
        elif self.series_length < 100:
            self.max_window = 44
        else:
            self.max_window = 84

        # bypass rule-of-thumb for ensemble_size entirely
        self.ensemble_size = self.r_max
        print(f"[FixedEnsemble size={self.ensemble_size}] "
              f"n={self.n_instances} len={self.series_length}")

        self.max_window = int(min(self.series_length, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window={self.min_window} is bigger than "
                f"max_window={self.max_window}, series length is {self.series_length}"
            )

        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, prefer="threads")(
            delayed(_parallel_fit)(
                i,
                XX,
                y.copy(),
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.lower_bounding,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.remove_repeat_words,
                self.sections,
                self.random_state,
            )
            for i in range(self.ensemble_size)
        )

        sfa_words = []
        for (words, transformer) in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        if type(sfa_words[0]) is np.ndarray:
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            all_words = hstack(sfa_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))
        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[1]
        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self


class WEASEL_V2_TFIDF(WEASEL_V2):
    """WEASEL 2.0 + TF-IDF weighting (use_idf=True, sublinear_tf=False)."""

    def _fit(self, X, y):
        super()._fit(X, y)
        all_words = self._transform_words(X)

        self._tfidf = TfidfTransformer(use_idf=True, sublinear_tf=False)
        all_words_tfidf = self._tfidf.fit_transform(all_words)

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


class WEASEL_V2_NoDiff(WEASEL_V2):
    """
    WEASEL 2.0 without applying dilation to first-order differences.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Try to disable first differences via known attribute names.
        # If the parent uses a different name, update this.
        if hasattr(self, "use_first_differences"):
            self.use_first_differences = False
        if hasattr(self, "first_difference"):
            self.first_difference = False
