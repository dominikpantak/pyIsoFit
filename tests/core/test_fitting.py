import unittest
import pytest
import pandas as pd
import unittest

import pyIsoFit
import pyIsoFit.core.exceptions as pfEx

params = {
            'key_uptakes': ['loading'],
            'key_pressures': ['pressure'],
            'temps': [20],
            'compname': 'N2'
        }

pressure = [1, 2, 3, 4, 5, 6, 7]
loading = [1, 2, 3, 4, 5, 6, 7]

df = pd.DataFrame({
    'pressure': pressure,
    'loading': loading
})

class TestFitting(unittest.TestCase):
    """Test the IsothermFit class"""
    def test_isotherm_fit(self):
        # Regular creation
        pyIsoFit.IsothermFit(
            df=df,
            model='Henry',
            **params
        )

        # Missing uptake
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=df,
                key_pressures=['pressure'],
                key_uptakes=None,
                temps=[20],
                model='Henry'
            )

        # Missing pressure
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=df,
                key_pressures=None,
                key_uptakes=['loading'],
                temps=[20],
                model='Henry'
            )

        # Missing df
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=None,
                key_pressures=['pressure'],
                key_uptakes=['loading'],
                temps=[20],
                model='Henry'
            )

        # Missing temps
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=df,
                key_pressures=['pressure'],
                key_uptakes=['loading'],
                temps=None,
                model='Henry'
            )

        # Wrong model
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=df,
                key_pressures=['pressure'],
                key_uptakes=['loading'],
                temps=None,
                model=''
            )

        # List lengths are different
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=df,
                key_pressures=['pressure'],
                key_uptakes=['loading'],
                temps=['20', '40'],
                model='Henry'
            )

        # df is a list but dsl is not the model
        with pytest.raises(pfEx.ParameterError):
            pyIsoFit.IsothermFit(
                df=[df],
                key_pressures=['pressure'],
                key_uptakes=['loading'],
                temps=['20'],
                model='Henry'
            )
    def test_fit(self):

        # Regular creation
        test_fit1 = pyIsoFit.IsothermFit(
            df=df,
            model='langmuir',
            **params
        )

        # df is a list but dsl is not the model
        with pytest.raises(pfEx.ParameterError):
            test_fit1.fit(
                guess={'sausage': [0.2]}
            )

