import typing as t

import numpy as np
from scipy.stats import lognorm

corrosion_type = t.Literal['carbonation', 'chloride']
exposure_class = t.Literal[
    'sheltered',
    'unsheltered',
    'wet',
    'cyclic_dry_wet',
    'airborn_seawater',
    'submerged',
    'tidal_zone',
]
exposure_class_corrosion_type = {
    'carbonation': ['sheltered', 'unsheltered'],
    'chloride': [
        'wet',
        'cyclic_dry_wet',
        'airborn_seawater',
        'submerged',
        'tidal_zone',
    ],
}


def _validate_corrosion_exposure(
    corrosion_type: corrosion_type, exposure_class: exposure_class
) -> bool:
    """Returns True if corrosion_type and exposure_class are valid."""
    corr_type = corrosion_type.lower()
    exposure = exposure_class.lower()
    if corr_type not in ['carbonation', 'chloride']:
        return False
    return exposure in exposure_class_corrosion_type[corr_type]


_table_30_1_6 = {
    'carbonation': {
        'sheltered': {'mean': 2, 'std': 3},
        # the table shows "t", but 7 is correct
        'unsheltered': {'mean': 5, 'std': 7},
    },
    'chloride': {
        'wet': {'mean': 4, 'std': 6},
        'cyclic_dry_wet': {'mean': 30, 'std': 40},
        'airborn_seawater': {'mean': 30, 'std': 40},
        'submerged': {'mean': 4, 'std': 7},
        'tidal_zone': {'mean': 50, 'std': 100},
    },
}


def pcorr_rep(
    corrosion_type: corrosion_type, exposure_class: exposure_class
) -> dict:
    """Returns corrosion rate properties in micrometer/year.

    This implements MC 2020 table 30.1-6a and 30.1-6b.
    The return is a dictionary containing mean and standard deviation of pcorr
    in micrometer/year.

    Note:
        There is a typo in MC2020 Table 30.1-6a. In unshelterd conditions, the
        pcorr standard deviation is indicated as "t" while it should be equal
        to 7 micrometer/year according to fib Bulletin 111.

    Args:
        corrosion_type (str): The type of corrosion. Can be either
            "carbonation" or "chloride".
        exposure_class (str): The exposure class. Can be one of the following
            for "carbonation": "sheltered", "unsheltered".
            Can be one of the following for "chloride":
            "wet", "cyclic_dry_wet", "airborn_seawater", "submerged",
            "tidal_zone".

    Returns:
        dict: A dictionary containing the mean and standard deviation of the
            corrosion rate in micrometer/year.

    Raises:
        ValueError: if the corrosion type is not valid.
        ValueError: if the exposure class is not valid for the relative
            corrosion type.
    """
    if not _validate_corrosion_exposure(corrosion_type, exposure_class):
        raise ValueError(
            'Invalid corrosion_type or exposure_class.\n'
            'corrosion_type must be either "carbonation" or "chloride".\n'
            'exposure_class must be one of the following for "carbonation": '
            '"sheltered", "unsheltered".\n'
            'exposure_class must be one of the following for "chloride": '
            '"wet", "cyclic_dry_wet", "airborn_seawater", "submerged", '
            '"tidal_zone".'
        )
    return _table_30_1_6[corrosion_type.lower()][exposure_class.lower()]


def pcorr_fractile(
    corrosion_type: corrosion_type,
    exposure_class: exposure_class,
    fractile: float = 0.5,
) -> float:
    """Returns the fractile of the corrosion rate in micrometer/year.

    This functions uses data from MC 2020 table 30.1-6a and 30.1-6b to
    calculate the corrosion rate at a given fractile. The return is a float
    representing the corrosion rate in micrometer/year.

    Note: according to fib Bulletin 111, the corrosion rate can be assumed to
        follow a lognormal distribution.

    Args:
        corrosion_type (str): The type of corrosion. Can be either
            "carbonation" or "chloride".
        exposure_class (str): The exposure class. Can be one of the following
            for "carbonation": "sheltered", "unsheltered".
            Can be one of the following for "chloride":
            "wet", "cyclic_dry_wet", "airborn_seawater", "submerged",
            "tidal_zone".
        fractile (Optional[float]): The fractile for which to calculate the
            corrosion rate. The default value is 0.5, which corresponds to the
            median corrosion rate. The value should be in the range ]0,1[

    Returns:
        float: The corrosion rate at the specified fractile in micrometer/year.

    Raises:
        ValueError: if the corrosion type is not valid.
        ValueError: if the exposure class is not valid for the relative
            corrosion type.
        ValueError: if the fractile value is not valid.
    """
    if fractile <= 0 or fractile >= 1:
        raise ValueError(
            'The value of fractile is not valid, use a value between 0 and 1.'
        )
    pcorr_dic = pcorr_rep(corrosion_type, exposure_class)

    # Lognormal distribution
    s = np.log(1 + pcorr_dic['std'] ** 2 / pcorr_dic['mean'] ** 2) ** 0.5
    scale = (
        pcorr_dic['mean'] ** 2
        / (pcorr_dic['mean'] ** 2 + pcorr_dic['std'] ** 2) ** 0.5
    )
    lognorm_dist = lognorm(s=s, scale=scale)
    return lognorm_dist.ppf(fractile)


def kc(fc: float) -> float:
    """Compute kc reducing compressive strength factor.

    MC2020 30.1.10.3.2.1

    It does depend only on fc and not on corrosion level, even if it is valid
    for low to moderate corrosion values.

    Moderate and low corrosion levels have been defined to apply approximately
    for medium bar diameters with 5% weight loss and 0.25 depth of corrosion.

    Args:
        fc (float): Compressive strength of concrete in MPa.

    Returns:
        float: The reducing compressive strength factor kc.

    Raises:
        ValueError: if fc is not positive.
    """
    if fc <= 0:
        raise ValueError('fc must be a positive value.')
    eta_fc = min((30 / fc) ** (1 / 3.0), 1)
    return 0.75 * eta_fc


# def calculate_minimum_area_after_corrosion(
#     uncorroded_area: float,
#     pitting_factor: t.Optional[float] = 1,
#     mass_loss: t.Optional[float] = None,
#     velocity_of_corrosion: t.Optional[float] = None,
#     time_of_corrosion: t.Optional[float] = None,
# ) -> float:
#     """A function to calculate the minimum residual steel area after corrosion.
#     The function allows two alternative approaches: (i) using the total mass loss,
#     or (ii) using the velocity of corrosion together with the time of corrosion.
#     A pitting factor is used to relate the maximum pit depth to the average pit depth.
#     Actually (02/12/2025) only MC2020 is implemented, see chapter 30.1.11.3.5.
#     Function is valid for round bars!.

#     Keyword Arguments:
#         uncorroded_area (float): Original (uncorroded) cross-sectional area of the rebar [mm²].
#         pitting_factor (float): Ratio between maximum pit depth and average pit depth
#             (default: 1). Must be ≥ 1.
#         mass_loss (float): Fractional mass loss (0–1). If provided, velocity_of_corrosion
#             and time_of_corrosion must be None. (default: None)
#         velocity_of_corrosion (float): Corrosion rate [μm/yr]. Must be ≥ 0. Used together
#             with time_of_corrosion. If provided, mass_loss must be None. (default: None)
#         time_of_corrosion (float): Time of corrosion [years]. Must be ≥ 0. Used with
#             velocity_of_corrosion. (default: None)

#     Return:
#         corroded_minimum_area (float): Minimum residual area of the corroded rebar [mm²],
#             computed assuming axisymmetric corrosion with maximum pit depth defined by
#             `pitting_factor * average_pit_depth`.

#     Raises:
#         ValueError: if both mass_loss and (velocity_of_corrosion + time_of_corrosion)
#             are provided simultaneously.
#         ValueError: if only one between velocity_of_corrosion and time_of_corrosion is provided.
#         ValueError: if pitting_factor < 1.
#         ValueError: if uncorroded_area < 0.
#         ValueError: if mass_loss is not in the range [0, 1].
#         ValueError: if velocity_of_corrosion < 0 or time_of_corrosion < 0.
#         ValueError: if the computed minimum residual radius becomes negative.
#     """
#     import math
#     from math import sqrt

#     uncorroded_diameter = sqrt(uncorroded_area / math.pi) * 2

#     # Input data validation
#     if (mass_loss is not None) and (
#         velocity_of_corrosion is not None or time_of_corrosion is not None
#     ):
#         raise ValueError(
#             'Too many input arguments have been given to the function. '
#             'Either use mass_loss or velocity_of_corrosion + time_of_corrosion.'
#         )

#     if (velocity_of_corrosion is not None and time_of_corrosion is None) or (
#         velocity_of_corrosion is None and time_of_corrosion is not None
#     ):
#         raise ValueError(
#             'velocity_of_corrosion or time_of_corrosion is missing.'
#         )

#     if pitting_factor < 1:
#         raise ValueError(
#             'The pitting_factor must be greater than or equal to 1.'
#         )

#     if uncorroded_area < 0:
#         raise ValueError('The uncorroded_area must be non-negative.')

#     if mass_loss is not None and (mass_loss < 0 or mass_loss > 1):
#         raise ValueError('mass_loss must be between 0 and 1.')

#     if velocity_of_corrosion is not None and (velocity_of_corrosion < 0):
#         raise ValueError('velocity_of_corrosion must be non-negative.')

#     if time_of_corrosion is not None and (time_of_corrosion < 0):
#         raise ValueError('time_of_corrosion must be non-negative.')

#     # Calculate corroded area if mass_loss is given
#     if mass_loss is not None:
#         corroded_average_area = uncorroded_area * (1 - mass_loss)
#         corroded_average_pit = uncorroded_diameter / 2 - sqrt(
#             corroded_average_area / math.pi
#         )
#         corroded_maximum_pit = pitting_factor * corroded_average_pit
#         corroded_minimum_area = (
#             uncorroded_diameter / 2 - corroded_maximum_pit
#         ) ** 2 * math.pi
#         if (uncorroded_diameter / 2 - corroded_maximum_pit) < 0:
#             raise ValueError(
#                 'The combination of mass_loss and pitting_factor gave a corroded_minimum_area'
#                 'lower than 0. Consider changing these values.'
#             )
#         return corroded_minimum_area

#     # Calculate corroded area if time_of_corrosion and velocity_of_corrosion are given
#     if velocity_of_corrosion is not None and time_of_corrosion is not None:
#         velocity_of_corrosion = (
#             velocity_of_corrosion / 1000
#         )  # convert from micrometers/year to millimeters/year
#         corroded_average_pit = velocity_of_corrosion * time_of_corrosion
#         if corroded_average_pit > uncorroded_diameter:
#             raise ValueError(
#                 'The combination of velocity_of_corrosion and time_of_corrosion gave a corroded_minimum_area'
#                 'lower than 0. Consider changing these values.'
#             )
#         corroded_maximum_pit = pitting_factor * corroded_average_pit
#         corroded_minimum_area = (
#             uncorroded_diameter / 2 - corroded_maximum_pit
#         ) ** 2 * math.pi
#         if uncorroded_diameter / 2 - corroded_maximum_pit < 0:
#             raise ValueError(
#                 'The combination of velocity_of_corrosion and time_of_corrosion and pitting_factor'
#                 'gave a corroded_minimum_area lower than 0. Consider changing these values.'
#             )
#         return corroded_minimum_area
#     return None
