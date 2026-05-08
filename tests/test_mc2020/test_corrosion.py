"""Tests for the functions in corrosion."""

from math import isclose

import numpy as np
import pytest
from scipy.stats import norm

from structuralcodes.codes.mc2020._corrosion import (
    kc,
    pcorr_fractile,
    pcorr_rep,
)


@pytest.mark.parametrize(
    'corrosion_type, exposure_class, expected',
    [
        ('carbonation', 'sheltered', (2, 3)),
        ('carbonation', 'unsheltered', (5, 7)),
        ('chloride', 'wet', (4, 6)),
        ('chloride', 'cyclic_dry_wet', (30, 40)),
        ('chloride', 'airborn_seawater', (30, 40)),
        ('chloride', 'submerged', (4, 7)),
        ('chloride', 'tidal_zone', (50, 100)),
        ('Chloride', 'Tidal_zone', (50, 100)),
    ],
)
def test_pcorr_dict(corrosion_type, exposure_class, expected):
    """Test pcorr_rep with valid input."""
    pcorr_dic = pcorr_rep(corrosion_type, exposure_class)
    assert pcorr_dic['mean'] == expected[0]
    assert pcorr_dic['std'] == expected[1]


@pytest.mark.parametrize(
    'corrosion_type, exposure_class',
    [
        ('chloride', 'sheltered'),
        ('carbonatio', 'unsheltered'),
        ('chloride', 'wett'),
        ('chloride', 'sheltered'),
        ('chloride', 'sunmerged'),
    ],
)
def test_pcorr_dict_invalid(corrosion_type, exposure_class):
    """Test pcorr_rep with valid input."""
    with pytest.raises(ValueError):
        pcorr_rep(corrosion_type, exposure_class)


@pytest.mark.parametrize(
    'fractile',
    [
        0.5,
        0.05,
        0.95,
        0.16,
        0.84,
    ],
)
@pytest.mark.parametrize(
    'corrosion_type, exposure_class',
    [
        ('carbonation', 'sheltered'),
        ('carbonation', 'unsheltered'),
        ('chloride', 'wet'),
        ('chloride', 'cyclic_dry_wet'),
        ('chloride', 'airborn_seawater'),
        ('chloride', 'submerged'),
        ('chloride', 'tidal_zone'),
    ],
)
def test_pcorr_fractile(corrosion_type, exposure_class, fractile):
    """Test pcorr_fractile with valid input."""
    # Evaluate expected value
    pcorr_dic = pcorr_rep(
        corrosion_type=corrosion_type, exposure_class=exposure_class
    )
    scale = (
        pcorr_dic['mean'] ** 2
        / (pcorr_dic['mean'] ** 2 + pcorr_dic['std'] ** 2) ** 0.5
    )
    median = scale
    mu = np.log(median)  # median = scale = exp(mu)
    s = (
        np.log(1 + pcorr_dic['std'] ** 2 / pcorr_dic['mean'] ** 2) ** 0.5
    )  # sigma = s
    z = norm.ppf(fractile)
    expected = np.exp(mu + z * s)
    # Act
    assert isclose(
        pcorr_fractile(
            corrosion_type=corrosion_type,
            exposure_class=exposure_class,
            fractile=fractile,
        ),
        expected,
    )


@pytest.mark.parametrize(
    'corrosion_type, exposure_class',
    [
        ('chloride', 'sheltered'),
        ('carbonatio', 'unsheltered'),
        ('chloride', 'wett'),
        ('chloride', 'sheltered'),
        ('chloride', 'sunmerged'),
    ],
)
def test_pcorr_fractile_invalid(corrosion_type, exposure_class):
    """Test pcorr_fractile with invalid  corrosion type and/or exposure."""
    with pytest.raises(ValueError):
        (
            pcorr_fractile(
                corrosion_type=corrosion_type,
                exposure_class=exposure_class,
                fractile=0.5,
            ),
        )


@pytest.mark.parametrize(
    'fractile',
    [
        0.0,
        -0.1,
        50,
        1.0,
        1.2,
        95.0,
    ],
)
@pytest.mark.parametrize(
    'corrosion_type, exposure_class',
    [
        ('carbonation', 'sheltered'),
        ('carbonation', 'unsheltered'),
        ('chloride', 'wet'),
        ('chloride', 'cyclic_dry_wet'),
        ('chloride', 'airborn_seawater'),
        ('chloride', 'submerged'),
        ('chloride', 'tidal_zone'),
    ],
)
def test_pcorr_fractile_invalid_fractile(
    corrosion_type, exposure_class, fractile
):
    """Test pcorr_fractile with invalid fractile."""
    with pytest.raises(ValueError):
        (
            pcorr_fractile(
                corrosion_type=corrosion_type,
                exposure_class=exposure_class,
                fractile=fractile,
            ),
        )


@pytest.mark.parametrize(
    'fc, expected',
    [
        (20, 0.75),
        (25, 0.75),
        (30, 0.75),
        (35, 0.712435688694747),
        (40, 0.681420222312052),
        (45, 0.655185348552224),
        (50, 0.632574498976312),
    ],
)
def test_kc(fc, expected):
    """Test kc with valid input."""
    assert isclose(kc(fc), expected)


@pytest.mark.parametrize(
    'fc',
    [
        (0),
        (-20),
    ],
)
def test_kc_invalid(fc):
    """Test kc with invalid input."""
    with pytest.raises(ValueError):
        kc(fc)


# # Should return the remaining area after corrosion.
# @pytest.mark.parametrize(
#     'mass_loss, pitting_factor, expected',
#     [
#         (0, 1, InitialArea),
#         (0, 2, InitialArea),
#         (0.5, 1, 0.5 * InitialArea),
#         (0.5, 2, (2 * 0.70710678118 - 1) ** 2 * InitialArea),
#         (0.75, 1, 0.25 * InitialArea),
#         (1, 1, 0),
#     ],
# )
# def test_minimum_area_after_corrosion_given_mass_loss_and_pitting_factor(
#     mass_loss, pitting_factor, expected
# ):
#     InitialArea = 8 * 8 * 3.141592
#     Minimum_area_after_corrosion = calculate_minimum_area_after_corrosion(
#         uncorroded_area=InitialArea,
#         pitting_factor=pitting_factor,
#         mass_loss=mass_loss,
#     )
#     assert (
#         abs(Minimum_area_after_corrosion - expected) <= expected * 0.00000001
#     )


# # Should return the remaining area after corrosion.
# @pytest.mark.parametrize(
#     'velocity_of_corrosion,time_of_corrosion, pitting_factor, expected',
#     [
#         (0, 0, 1, InitialArea),
#         (0, 0, 2, InitialArea),
#         (100, 10, 1, (7**2 / 8**2) * InitialArea),
#         (100, 10, 2, (6**2 / 8**2) * InitialArea),
#         (100, 40, 1, (4**2 / 8**2) * InitialArea),
#     ],
# )
# def test_minimum_area_after_corrosion_given_velocity_of_corrosion_and_time_of_corrosion(
#     velocity_of_corrosion, time_of_corrosion, pitting_factor, expected
# ):
#     InitialArea = 8 * 8 * 3.141592
#     Minimum_area_after_corrosion = calculate_minimum_area_after_corrosion(
#         uncorroded_area=InitialArea,
#         pitting_factor=pitting_factor,
#         velocity_of_corrosion=velocity_of_corrosion,
#         time_of_corrosion=time_of_corrosion,
#     )
#     assert abs(Minimum_area_after_corrosion - expected) <= expected * 0.0001


# # Should raise different kind of errors (wrong combinations of input values or negative area after corrosion).
# @pytest.mark.parametrize(
#     'velocity_of_corrosion, time_of_corrosion, mass_loss, pitting_factor',
#     [
#         (100, 10, 0.5, 1),
#         (100, None, 0.5, 1),
#         (None, 10, 0.5, 1),
#         (100, 100, None, 1),
#         (100, 50, None, 2),
#         (None, None, 1.5, 1),
#         (None, None, 0.9, 2),
#     ],
# )
# def test_wrong_combinations_or_negative_area_after_corrosion(
#     velocity_of_corrosion, time_of_corrosion, pitting_factor, mass_loss
# ):
#     InitialArea = 8 * 8 * 3.141592
#     with pytest.raises(Exception):
#         Minimum_area_after_corrosion = calculate_minimum_area_after_corrosion(
#             mass_loss=mass_loss,
#             uncorroded_area=InitialArea,
#             pitting_factor=pitting_factor,
#             velocity_of_corrosion=velocity_of_corrosion,
#             time_of_corrosion=time_of_corrosion,
#         )
