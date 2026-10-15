"""Offline regression cases using real estimator public APIs, without mocks."""

import pandas as pd
import pytest
from category_encoders import OneHotEncoder, OrdinalEncoder, SummaryEncoder, TargetEncoder
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

ENCODERS = [OrdinalEncoder, OneHotEncoder, TargetEncoder, SummaryEncoder]
TRANSITIONS = [
    (None, ['a']),
    (None, []),
    (['a'], None),
    (['a'], 'all'),
    ('all', ['a']),
    ('all', None),
    (None, 'ALL'),
    (['a'], ['b']),
]
X = pd.DataFrame({'a': ['x', 'y'] * 4, 'b': ['p', 'q', 'q', 'p'] * 2, 'n': [10, 20] * 4})
y = pd.Series([0.0, 1.0, 2.0, 3.0] * 2)


@pytest.mark.parametrize('encoder_cls', ENCODERS)
@pytest.mark.parametrize('initial,selected', TRANSITIONS)
def test_set_params_matches_constructor(encoder_cls, initial, selected):
    """Column changes must behave like construction with the same selection."""
    expected = encoder_cls(cols=selected).fit_transform(X, y)
    enc = encoder_cls(cols=initial)
    assert enc.set_params(cols=selected) is enc
    actual = enc.fit_transform(X, y)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize('encoder_cls', ENCODERS)
def test_pipeline_nested_set_params(encoder_cls):
    """Pipeline parameter forwarding must preserve the requested selection."""
    expected = Pipeline([('encode', encoder_cls(cols=['a']))]).fit_transform(X, y)
    actual = (
        Pipeline([('encode', encoder_cls())]).set_params(encode__cols=['a']).fit_transform(X, y)
    )
    assert_frame_equal(actual, expected)


def test_gridsearch_evaluates_requested_columns():
    """Grid search must evaluate each requested column selection."""
    # Scorer observes fitted columns, so no downstream model obscures the bug.
    search = GridSearchCV(
        OneHotEncoder(),
        {'cols': [['a'], ['a', 'b']]},
        scoring=lambda encoder, X, y: len(encoder.cols),
        cv=2,
        refit=False,
    )
    search.fit(X, y)
    assert search.cv_results_['mean_test_score'].tolist() == [1.0, 2.0]


@pytest.mark.parametrize('encoder_cls', ENCODERS)
@pytest.mark.parametrize('initial', [None, 'all'])
def test_unrelated_set_params_preserves_automatic_refit(encoder_cls, initial):
    """Unrelated parameters must preserve automatic selection on repeated fit."""
    enc = encoder_cls(cols=initial).fit(X, y)
    X2 = X.rename(columns={'a': 'c', 'b': 'd', 'n': 'm'})
    enc.set_params(verbose=0)
    assert_frame_equal(enc.fit_transform(X2, y), encoder_cls(cols=initial).fit_transform(X2, y))


@pytest.mark.parametrize('encoder_cls', ENCODERS)
@pytest.mark.parametrize('initial,selected', TRANSITIONS)
def test_change_selection_after_fit(encoder_cls, initial, selected):
    """A fitted encoder must honor a newly specified column selection."""
    enc = encoder_cls(cols=initial).fit(X, y)
    enc.set_params(cols=selected)
    assert_frame_equal(enc.fit_transform(X, y), encoder_cls(cols=selected).fit_transform(X, y))


@pytest.mark.parametrize('encoder_cls', ENCODERS)
@pytest.mark.parametrize('selected', [None, 'all', ['a'], []])
def test_clone_after_set_params(encoder_cls, selected):
    """Clones and explicitly reconfigured estimators must fit identically."""
    enc = encoder_cls(cols=['b']).set_params(cols=selected)
    cloned = clone(enc)
    assert_frame_equal(enc.fit_transform(X, y), cloned.fit_transform(X, y))


@pytest.mark.parametrize('encoder_cls', ENCODERS)
def test_set_params_preserves_sklearn_contract(encoder_cls):
    """Preserve parameter validation, updates, and fluent return values."""
    enc = encoder_cls()
    assert enc.set_params() is enc
    assert enc.set_params(verbose=2, return_df=False) is enc
    assert enc.get_params()['verbose'] == 2
    assert enc.get_params()['return_df'] is False
    with pytest.raises(ValueError, match='Invalid parameter'):
        enc.set_params(nonexistent_parameter=True)
