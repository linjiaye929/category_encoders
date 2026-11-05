"""Preserve categorical values when restoring dtypes for arraylike transforms."""

import numpy as np
import pandas as pd
import pytest
from category_encoders import OneHotEncoder, OrdinalEncoder, QuantileEncoder, TargetEncoder


@pytest.mark.parametrize(
    'encoder_class', [OrdinalEncoder, OneHotEncoder, TargetEncoder, QuantileEncoder]
)
@pytest.mark.parametrize('ordered', [False, True])
def test_array_unknown_category_policy(encoder_class, ordered):
    """Restoring fitted categories must not turn an unseen value into missing data."""
    X = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a', 'b'], ordered=ordered)})
    encoder = encoder_class(cols=['cat'], handle_unknown='error').fit(X, [0, 1, 0, 1])
    unseen = pd.DataFrame({'cat': ['new']})
    for data in (unseen, unseen.to_numpy()):
        with pytest.raises(ValueError):
            encoder.transform(data)


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('input_kind', ['array', 'list'])
@pytest.mark.parametrize(
    'values, expected',
    [
        (['new', None, np.nan, pd.NA, 'b', 'unused'], [-1, -2, -2, -2, 1, -1]),
        (['new', 'other'], [-1, -1]),
        ([None, np.nan, pd.NA], [-2, -2, -2]),
        (['unused'], [-1]),
        (['b', 'a'], [1, 2]),
    ],
)
def test_unknown_and_missing_values_remain_distinct(ordered, input_kind, values, expected):
    """Observed, unused, unknown, and missing values retain their existing ordinal meaning."""
    dtype = pd.CategoricalDtype(categories=['b', 'a', 'unused'], ordered=ordered)
    X = pd.DataFrame({0: pd.Series(['b', 'a', 'b', 'a'], dtype=dtype)})
    encoder = OrdinalEncoder(cols=[0]).fit(X)
    testing = pd.DataFrame({0: values})
    data = testing.to_numpy() if input_kind == 'array' else testing.to_numpy().tolist()
    np.testing.assert_array_equal(encoder.transform(testing)[0], expected)
    np.testing.assert_array_equal(encoder.transform(data)[0], expected)
    assert encoder.feature_dtypes_in_[0] == dtype
    assert encoder.feature_dtypes_in_[0].categories.tolist() == ['b', 'a', 'unused']


@pytest.mark.parametrize('encoder_class', [TargetEncoder, QuantileEncoder])
def test_supervised_array_unknown_and_missing_fills(encoder_class):
    """The supervised transform path must retain distinct unknown and missing fills."""
    X = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a', 'b'])})
    encoder = encoder_class(cols=['cat'], handle_unknown=-91, handle_missing=-81)
    encoder.fit(X, [0, 1, 0, 1])
    testing = pd.DataFrame({'cat': ['new', None, np.nan, pd.NA, 'a']})
    expected = encoder.transform(testing)
    np.testing.assert_array_equal(expected['cat'].iloc[:4], [-91, -81, -81, -81])
    pd.testing.assert_frame_equal(encoder.transform(testing.to_numpy()), expected)


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('values', [['new', 'x'], ['x', 'y'], [None, np.nan]])
def test_array_categorical_passthrough_values_and_dtypes(ordered, values):
    """Keep passthrough values, restoring the fitted categorical dtype only when lossless."""
    dtype = pd.CategoricalDtype(categories=['y', 'x', 'unused'], ordered=ordered)
    X = pd.DataFrame({'encoded': ['a', 'b'], 'untouched': pd.Series(['x', 'y'], dtype=dtype)})
    original = X.copy(deep=True)
    encoder = OrdinalEncoder(cols=['encoded']).fit(X)
    mapping = encoder.mapping[0]['mapping'].copy(deep=True)
    testing = pd.DataFrame({'encoded': ['a', 'b'], 'untouched': values})
    data = testing.to_numpy()
    saved_data = data.copy()
    actual = encoder.transform(data)
    expected = encoder.transform(testing)
    pd.testing.assert_series_equal(
        actual['untouched'].astype(object), expected['untouched'].astype(object)
    )
    if 'new' in values:
        assert actual['untouched'].dtype == object
    else:
        assert actual['untouched'].dtype == dtype
    assert encoder.feature_dtypes_in_[1] == dtype
    assert encoder.feature_dtypes_in_[1].categories.tolist() == ['y', 'x', 'unused']
    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_frame_equal(pd.DataFrame(data), pd.DataFrame(saved_data))
    pd.testing.assert_series_equal(encoder.mapping[0]['mapping'], mapping)
    # A transform containing unseen values must not alter later dtype restoration.
    assert encoder.transform(X.to_numpy())['untouched'].dtype == dtype


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('unseen', [False, True])
def test_array_duplicate_categorical_passthrough_columns(ordered, unseen):
    """Duplicate passthrough labels must preserve values without an ambiguous mask."""
    dtype = pd.CategoricalDtype(categories=['y', 'x', 'unused'], ordered=ordered)
    X = pd.concat(
        [
            pd.Series(['a', 'b'], name='encoded'),
            pd.Series(['x', 'y'], name='dup', dtype=dtype),
            pd.Series(['y', 'x'], name='dup', dtype=dtype),
        ],
        axis=1,
    )
    original = X.copy(deep=True)
    encoder = OrdinalEncoder(cols=['encoded']).fit(X)
    mapping = encoder.mapping[0]['mapping'].copy(deep=True)
    data = X.to_numpy()
    if unseen:
        data[0, 1] = 'new'
    saved_data = data.copy()
    expected = encoder.transform(pd.DataFrame(data, columns=X.columns))
    actual = encoder.transform(data)
    pd.testing.assert_frame_equal(actual.astype(object), expected.astype(object))
    assert encoder.feature_dtypes_in_[1:] == [dtype, dtype]
    assert encoder.feature_dtypes_in_[1].categories.tolist() == ['y', 'x', 'unused']
    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_frame_equal(pd.DataFrame(data), pd.DataFrame(saved_data))
    pd.testing.assert_series_equal(encoder.mapping[0]['mapping'], mapping)
    known = encoder.transform(X.to_numpy())
    assert known.dtypes.iloc[1:].tolist() == [dtype, dtype]
