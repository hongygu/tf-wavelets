"""
The 'dwtcoeffs' module contains predefined wavelets, as well as the classes necessary to
create more user-defined wavelets.

Wavelets are defined by the Wavelet class. A Wavelet object mainly consists of four Filter
objects (defined by the Filter class) representing the decomposition and reconstruction
low pass and high pass filters.

Examples:
    You can define your own wavelet by creating four filters, and combining them to a wavelet:

    >>> decomp_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)
    >>> decomp_hp = Filter([1 / np.sqrt(2), -1 / np.sqrt(2)], 1)
    >>> recon_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)
    >>> recon_hp = Filter([-1 / np.sqrt(2), 1 / np.sqrt(2)], 1)
    >>> haar = Wavelet(decomp_lp, decomp_hp, recon_lp, recon_hp)

"""

import numpy as np
import tensorflow as tf
from utils import adapt_filter, to_tf_mat


class Filter:
    """
    Class representing a filter.

    Attributes:
        coeffs (tf.constant):      Filter coefficients
        zero (int):                Origin of filter (which index of coeffs array is
                                   actually indexed as 0).
        edge_matrices (iterable):  List of edge matrices, used for circular convolution.
                                   Stored as 3D TF tensors (constants).
    """


    def __init__(self, coeffs, zero):
        """
        Create a filter based on given filter coefficients

        Args:
            coeffs (np.ndarray):       Filter coefficients
            zero (int):                Origin of filter (which index of coeffs array is
                                       actually indexed as 0).
        """
        self.coeffs = tf.constant(adapt_filter(coeffs), dtype=tf.float32)

        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(self.coeffs)
        self._coeffs = coeffs.astype(np.float32)

        self.zero = zero

        self.edge_matrices = to_tf_mat(self._edge_matrices())


    def __getitem__(self, item):
        """
        Returns filter coefficients at requested indeces. Indeces are offset by the filter
        origin

        Args:
            item (int or slice):    Item(s) to get

        Returns:
            np.ndarray: Item(s) at specified place(s)
        """
        if isinstance(item, slice):
            return self._coeffs.__getitem__(
                slice(item.start + self.zero, item.stop + self.zero, item.step)
            )
        else:
            return self._coeffs.__getitem__(item + self.zero)


    def num_pos(self):
        """
        Number of positive indexed coefficients in filter, including the origin. Ie,
        strictly speaking it's the number of non-negative indexed coefficients.

        Returns:
            int: Number of positive indexed coefficients in filter.
        """
        return len(self._coeffs) - self.zero


    def num_neg(self):
        """
        Number of negative indexed coefficients, excluding the origin.

        Returns:
            int: Number of negative indexed coefficients
        """
        return self.zero


    def _edge_matrices(self):
        """Computes the submatrices needed at the ends for circular convolution.

        Returns:
            Tuple of 2d-arrays, (top-left, top-right, bottom-left, bottom-right).
        """
        if not isinstance(self._coeffs, np.ndarray):
            self._coeffs = np.array(self._coeffs)

        n, = self._coeffs.shape
        self._coeffs = self._coeffs[::-1]

        # Some padding is necesssary to keep the submatrices
        # from having having columns in common
        padding = max((self.zero, n - self.zero - 1))
        matrix_size = n + padding
        filter_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        negative = self._coeffs[
                   -(self.zero + 1):]  # negative indexed filter coeffs (and 0)
        positive = self._coeffs[
                   :-(self.zero + 1)]  # filter coeffs with strictly positive indeces

        # Insert first row
        filter_matrix[0, :len(negative)] = negative

        # Because -0 == 0, a length of 0 makes it impossible to broadcast
        # (nor is is necessary)
        if len(positive) > 0:
            filter_matrix[0, -len(positive):] = positive

        # Cycle previous row to compute the entire filter matrix
        for i in range(1, matrix_size):
            filter_matrix[i, :] = np.roll(filter_matrix[i - 1, :], 1)

        # TODO: Indexing not thoroughly tested
        num_pos = len(positive)
        num_neg = len(negative)
        top_left = filter_matrix[:num_pos, :(num_pos + num_neg - 1)]
        top_right = filter_matrix[:num_pos, -num_pos:]
        bottom_left = filter_matrix[-num_neg + 1:, :num_neg - 1]
        bottom_right = filter_matrix[-num_neg + 1:, -(num_pos + num_neg - 1):]

        # Indexing wrong when there are no negative indexed coefficients
        if num_neg == 1:
            bottom_left = np.zeros((0, 0), dtype=np.float32)
            bottom_right = np.zeros((0, 0), dtype=np.float32)

        return top_left, top_right, bottom_left, bottom_right


class TrainableFilter(Filter):
    """
    Class representing a trainable filter.

    Attributes:
        coeffs (tf.Variable):      Filter coefficients
        zero (int):                Origin of filter (which index of coeffs array is
                                   actually indexed as 0).
    """


    def __init__(self, initial_coeffs, zero, name=None):
        """
        Create a trainable filter initialized with given filter coefficients

        Args:
            initial_coeffs (np.ndarray):    Initial filter coefficients
            zero (int):                     Origin of filter (which index of coeffs array
                                            is actually indexed as 0).
            name (str):                     Optional. Name of tf variable created to hold
                                            the filter coeffs.
        """
        super().__init__(initial_coeffs, zero)

        self.coeffs = tf.Variable(
            initial_value=adapt_filter(initial_coeffs),
            trainable=True,
            name=name,
            dtype=tf.float32,
            constraint=tf.keras.constraints.max_norm(np.sqrt(2), [1, 2])
        )

        # Erase stuff that will be invalid once the filter coeffs has changed
        self._coeffs = [None]*len(self._coeffs)
        self.edge_matrices = None


class Wavelet:
    """
    Class representing a wavelet.

    Attributes:
        decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
        decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
        recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
        recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
    """


    def __init__(self, decomp_lp, decomp_hp, recon_lp, recon_hp):
        """
        Create a new wavelet based on specified filters

        Args:
            decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
            decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
            recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
            recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
        """
        self.decomp_lp = decomp_lp
        self.decomp_hp = decomp_hp
        self.recon_lp = recon_lp
        self.recon_hp = recon_hp


class TrainableWavelet(Wavelet):
    """
    Class representing a trainable wavelet

    Attributes:
        decomp_lp (TrainableFilter):    Filter coefficients for decomposition low pass filter
        decomp_hp (TrainableFilter):    Filter coefficients for decomposition high pass filter
        recon_lp (TrainableFilter):     Filter coefficients for reconstruction low pass filter
        recon_hp (TrainableFilter):     Filter coefficients for reconstruction high pass filter
    """


    def __init__(self, wavelet):
        """
        Create a new trainable wavelet initialized as specified wavelet

        Args:
            wavelet (Wavelet):          Starting point for the trainable wavelet
        """
        super().__init__(
            TrainableFilter(wavelet.decomp_lp._coeffs, wavelet.decomp_lp.zero),
            TrainableFilter(wavelet.decomp_hp._coeffs, wavelet.decomp_hp.zero),
            TrainableFilter(wavelet.recon_lp._coeffs, wavelet.recon_lp.zero),
            TrainableFilter(wavelet.recon_hp._coeffs, wavelet.recon_hp.zero)
        )


# Haar wavelet
haar = Wavelet(
    Filter(np.array([0.70710677, 0.70710677]), 1),
    Filter(np.array([-0.70710677, 0.70710677]), 0),
    Filter(np.array([0.70710677, 0.70710677]), 0),
    Filter(np.array([0.70710677, -0.70710677]), 1),
)

# Daubechies wavelets
db1 = haar
db2 = Wavelet(
    Filter(np.array([-0.12940952255092145,
                     0.22414386804185735,
                     0.836516303737469,
                     0.48296291314469025]), 3),
    Filter(np.array([-0.48296291314469025,
                     0.836516303737469,
                     -0.22414386804185735,
                     -0.12940952255092145]), 0),
    Filter(np.array([0.48296291314469025,
                     0.836516303737469,
                     0.22414386804185735,
                     -0.12940952255092145]), 0),
    Filter(np.array([-0.12940952255092145,
                     -0.22414386804185735,
                     0.836516303737469,
                     -0.48296291314469025]), 3)
)
db3 = Wavelet(
    Filter(np.array([0.035226291882100656,
                    -0.08544127388224149,
                    -0.13501102001039084,
                    0.4598775021193313,
                    0.8068915093133388,
                    0.3326705529509569]), 5),
    Filter(np.array([-0.3326705529509569,
                    0.8068915093133388,
                    -0.4598775021193313,
                    -0.13501102001039084,
                    0.08544127388224149,
                    0.035226291882100656]), 0),
    Filter(np.array([0.3326705529509569,
                    0.8068915093133388,
                    0.4598775021193313,
                    -0.13501102001039084,
                    -0.08544127388224149,
                    0.035226291882100656]), 0),
    Filter(np.array([0.035226291882100656,
                    0.08544127388224149,
                    -0.13501102001039084,
                    -0.4598775021193313,
                    0.8068915093133388,
                    -0.3326705529509569]), 5)
)
db4 = Wavelet(
    Filter(np.array([-0.010597401784997278,
                    0.032883011666982945,
                    0.030841381835986965,
                    -0.18703481171888114,
                    -0.02798376941698385,
                    0.6308807679295904,
                    0.7148465705525415,
                    0.23037781330885523]), 7),
    Filter(np.array([-0.23037781330885523,
                    0.7148465705525415,
                    -0.6308807679295904,
                    -0.02798376941698385,
                    0.18703481171888114,
                    0.030841381835986965,
                    -0.032883011666982945,
                    -0.010597401784997278]), 0),
    Filter(np.array([0.23037781330885523,
                    0.7148465705525415,
                    0.6308807679295904,
                    -0.02798376941698385,
                    -0.18703481171888114,
                    0.030841381835986965,
                    0.032883011666982945,
                    -0.010597401784997278]), 0),
    Filter(np.array([-0.010597401784997278,
                    -0.032883011666982945,
                    0.030841381835986965,
                    0.18703481171888114,
                    -0.02798376941698385,
                    -0.6308807679295904,
                    0.7148465705525415,
                    -0.23037781330885523]), 7)
)

db5 = Wavelet(
    Filter(np.array([0.003335725,
                    -0.012580752,
                    -0.00624149,
                    0.077571494,
                    -0.03224487,
                    -0.242294887,
                    0.138428146,
                    0.724308528,
                    0.60382927,
                    0.160102398 ]), 9),
    Filter(np.array([-0.160102398,
                    0.60382927,
                    -0.724308528,
                    0.138428146,
                    0.242294887,
                    -0.03224487,
                    -0.077571494,
                    -0.00624149,
                    0.012580752,
                    0.003335725]), 0),
    Filter(np.array([0.160102398,
                    0.60382927,
                    0.724308528,
                    0.138428146,
                    -0.242294887,
                    -0.03224487,
                    0.077571494,
                    -0.00624149,
                    -0.012580752,
                    0.003335725]), 0),
    Filter(np.array([0.003335725,
                    0.012580752,
                    -0.00624149,
                    -0.077571494,
                    -0.03224487,
                    0.242294887,
                    0.138428146,
                    -0.724308528,
                    0.60382927,
                    -0.160102398]), 9)
)

db6 = Wavelet(
    Filter(np.array([-0.001077301,
                    0.004777258,
                    0.000553842,
                    -0.031582039,
                    0.027522866,
                    0.097501606,
                    -0.129766868,
                    -0.226264694,
                    0.315250352,
                    0.751133908,
                    0.49462389,
                    0.111540743]), 11),
    Filter(np.array([-0.111540743,
                    0.49462389,
                    -0.751133908,
                    0.315250352,
                    0.226264694,
                    -0.129766868,
                    -0.097501606,
                    0.027522866,
                    0.031582039,
                    0.000553842,
                    -0.004777258,
                    -0.001077301]), 0),
    Filter(np.array([0.111540743,
                    0.49462389,
                    0.751133908,
                    0.315250352,
                    -0.226264694,
                    -0.129766868,
                    0.097501606,
                    0.027522866,
                    -0.031582039,
                    0.000553842,
                    0.004777258,
                    -0.001077301]), 0),
    Filter(np.array([-0.001077301,
                    -0.004777258,
                    0.000553842,
                    0.031582039,
                    0.027522866,
                    -0.097501606,
                    -0.129766868,
                    0.226264694,
                    0.315250352,
                    -0.751133908,
                    0.49462389,
                    -0.111540743]), 11)
)

db7 = Wavelet(
    Filter(np.array([0.000353714,
                    -0.001801641,
                    0.000429578,
                    0.012550999,
                    -0.016574542,
                    -0.038029937,
                    0.080612609,
                    0.071309219,
                    -0.224036185,
                    -0.143906004,
                    0.469782287,
                    0.729132091,
                    0.396539319,
                    0.077852054]), 13),
    Filter(np.array([-0.077852054,
                    0.396539319,
                    -0.729132091,
                    0.469782287,
                    0.143906004,
                    -0.224036185,
                    -0.071309219,
                    0.080612609,
                    0.038029937,
                    -0.016574542,
                    -0.012550999,
                    0.000429578,
                    0.001801641,
                    0.000353714]), 0),
    Filter(np.array([0.077852054,
                    0.396539319,
                    0.729132091,
                    0.469782287,
                    -0.143906004,
                    -0.224036185,
                    0.071309219,
                    0.080612609,
                    -0.038029937,
                    -0.016574542,
                    0.012550999,
                    0.000429578,
                    -0.001801641,
                    0.000353714]), 0),
    Filter(np.array([0.000353714,
                    0.001801641,
                    0.000429578,
                    -0.012550999,
                    -0.016574542,
                    0.038029937,
                    0.080612609,
                    -0.071309219,
                    -0.224036185,
                    0.143906004,
                    0.469782287,
                    -0.729132091,
                    0.396539319,
                    -0.077852054]), 13)
)

sy5 = Wavelet(
    Filter(np.array([0.027333068,
                    0.029519491,
                    -0.039134249,
                    0.199397534,
                    0.72340769,
                    0.633978963,
                    0.016602106,
                    -0.17532809,
                    -0.021101834,
                    0.019538883 ]), 9),
    Filter(np.array([-0.019538883,
                    -0.021101834,
                    0.17532809,
                    0.016602106,
                    -0.633978963,
                    0.72340769,
                    -0.199397534,
                    -0.039134249,
                    -0.029519491,
                    0.027333068]), 0),
    Filter(np.array([0.019538883,
                    -0.021101834,
                    -0.17532809,
                    0.016602106,
                    0.633978963,
                    0.72340769,
                    0.199397534,
                    -0.039134249,
                    0.029519491,
                    0.027333068]), 0),
    Filter(np.array([0.027333068,
                    -0.029519491,
                    -0.039134249,
                    -0.199397534,
                    0.72340769,
                    -0.633978963,
                    0.016602106,
                    0.17532809,
                    -0.021101834,
                    -0.019538883]), 9)
)

sy2 = Wavelet(
    Filter(np.array([-0.12940952255092145,
                    0.22414386804185735,
                    0.836516303737469,
                    0.48296291314469025]), 3),
    Filter(np.array([-0.48296291314469025,
                    0.836516303737469,
                    -0.22414386804185735,
                    -0.12940952255092145]), 0),
    Filter(np.array([0.48296291314469025,
                    0.836516303737469,
                    0.22414386804185735,
                    -0.12940952255092145]), 0),
    Filter(np.array([-0.12940952255092145,
                    -0.22414386804185735,
                    0.836516303737469,
                    -0.48296291314469025]), 3)
)

sy4 = Wavelet(
    Filter(np.array([-0.07576571478927333,
                    -0.02963552764599851,
                    0.49761866763201545,
                    0.8037387518059161,
                    0.29785779560527736,
                    -0.09921954357684722,
                    -0.012603967262037833,
                    0.0322231006040427]), 7),
    Filter(np.array([-0.0322231006040427,
                    -0.012603967262037833,
                    0.09921954357684722,
                    0.29785779560527736,
                    -0.8037387518059161,
                    0.49761866763201545,
                    0.02963552764599851,
                    -0.07576571478927333]), 0),
    Filter(np.array([0.0322231006040427,
                    -0.012603967262037833,
                    -0.09921954357684722,
                    0.29785779560527736,
                    0.8037387518059161,
                    0.49761866763201545,
                    -0.02963552764599851,
                    -0.07576571478927333]), 0),
    Filter(np.array([-0.07576571478927333,
                    0.02963552764599851,
                    0.49761866763201545,
                    -0.8037387518059161,
                    0.29785779560527736,
                    0.09921954357684722,
                    -0.012603967262037833,
                    -0.0322231006040427]), 7)
)

sy6 = Wavelet(
    Filter(np.array([0.015404109327027373,
                    0.0034907120842174702,
                    -0.11799011114819057,
                    -0.048311742585633,
                    0.4910559419267466,
                    0.787641141030194,
                    0.3379294217276218,
                    -0.07263752278646252,
                    -0.021060292512300564,
                    0.04472490177066578,
                    0.0017677118642428036,
                    -0.007800708325034148]), 11),
    Filter(np.array([0.007800708325034148,
                    0.0017677118642428036,
                    -0.04472490177066578,
                    -0.021060292512300564,
                    0.07263752278646252,
                    0.3379294217276218,
                    -0.787641141030194,
                    0.4910559419267466,
                    0.048311742585633,
                    -0.11799011114819057,
                    -0.0034907120842174702,
                    0.015404109327027373]), 0),
    Filter(np.array([-0.007800708325034148,
                    0.0017677118642428036,
                    0.04472490177066578,
                    -0.021060292512300564,
                    -0.07263752278646252,
                    0.3379294217276218,
                    0.787641141030194,
                    0.4910559419267466,
                    -0.048311742585633,
                    -0.11799011114819057,
                    0.0034907120842174702,
                    0.015404109327027373]), 0),
    Filter(np.array([0.015404109327027373,
                    -0.0034907120842174702,
                    -0.11799011114819057,
                    0.048311742585633,
                    0.4910559419267466,
                    -0.787641141030194,
                    0.3379294217276218,
                    0.07263752278646252,
                    -0.021060292512300564,
                    -0.04472490177066578,
                    0.0017677118642428036,
                    0.007800708325034148]), 11)
)



sy8 = Wavelet(
    Filter(np.array([-0.0033824159510061256,
                    -0.0005421323317911481,
                    0.03169508781149298,
                    0.007607487324917605,
                    -0.1432942383508097,
                    -0.061273359067658524,
                    0.4813596512583722,
                    0.7771857517005235,
                    0.3644418948353314,
                    -0.05194583810770904,
                    -0.027219029917056003,
                    0.049137179673607506,
                    0.003808752013890615,
                    -0.01495225833704823,
                    -0.0003029205147213668,
                    0.0018899503327594609]), 15),
    Filter(np.array([-0.0018899503327594609,
                    -0.0003029205147213668,
                    0.01495225833704823,
                    0.003808752013890615,
                    -0.049137179673607506,
                    -0.027219029917056003,
                    0.05194583810770904,
                    0.3644418948353314,
                    -0.7771857517005235,
                    0.4813596512583722,
                    0.061273359067658524,
                    -0.1432942383508097,
                    -0.007607487324917605,
                    0.03169508781149298,
                    0.0005421323317911481,
                    -0.0033824159510061256]), 0),
    Filter(np.array([0.0018899503327594609,
                    -0.0003029205147213668,
                    -0.01495225833704823,
                    0.003808752013890615,
                    0.049137179673607506,
                    -0.027219029917056003,
                    -0.05194583810770904,
                    0.3644418948353314,
                    0.7771857517005235,
                    0.4813596512583722,
                    -0.061273359067658524,
                    -0.1432942383508097,
                    0.007607487324917605,
                    0.03169508781149298,
                    -0.0005421323317911481,
                    -0.0033824159510061256]), 0),
    Filter(np.array([-0.0033824159510061256,
                    0.0005421323317911481,
                    0.03169508781149298,
                    -0.007607487324917605,
                    -0.1432942383508097,
                    0.061273359067658524,
                    0.4813596512583722,
                    -0.7771857517005235,
                    0.3644418948353314,
                    0.05194583810770904,
                    -0.027219029917056003,
                    -0.049137179673607506,
                    0.003808752013890615,
                    0.01495225833704823,
                    -0.0003029205147213668,
                    -0.0018899503327594609]), 15)
)


co1 = Wavelet(
    Filter(np.array([-0.015655728,
                    -0.07273262,
                    0.384864847,
                    0.85257202,
                    0.337897662,
                    -0.07273262]), 5),
    Filter(np.array([0.07273262,
                    0.337897662,
                    -0.85257202,
                    0.384864847,
                    0.07273262,
                    -0.015655728]), 0),
    Filter(np.array([-0.07273262,
                    0.337897662,
                    0.85257202,
                    0.384864847,
                    -0.07273262,
                    -0.015655728]), 0),
    Filter(np.array([-0.015655728,
                    0.07273262,
                    0.384864847,
                    -0.85257202,
                    0.337897662,
                    0.07273262]), 5)
)

co2 = Wavelet(
    Filter(np.array([-0.000720549,
                    -0.001823209,
                    0.005611435,
                    0.023680172,
                    -0.059434419,
                    -0.076488599,
                    0.417005184,
                    0.812723635,
                    0.386110067,
                    -0.067372555,
                    -0.041464937,
                    0.016387336]), 11),
    Filter(np.array([-0.016387336,
                    -0.041464937,
                    0.067372555,
                    0.386110067,
                    -0.812723635,
                    0.417005184,
                    0.076488599,
                    -0.059434419,
                    -0.023680172,
                    0.005611435,
                    0.001823209,
                    -0.000720549 ]), 0),
    Filter(np.array([0.016387336,
                    -0.041464937,
                    -0.067372555,
                    0.386110067,
                     0.812723635,
                    0.417005184,
                    -0.076488599 ,
                    -0.059434419,
                    0.023680172,
                    0.005611435,
                    -0.001823209 ,
                    -0.000720549]), 0),
    Filter(np.array([-0.000720549,
                    0.001823209,
                    0.005611435,
                    -0.023680172,
                    -0.059434419,
                    0.076488599,
                    0.417005184,
                    -0.812723635,
                    0.386110067,
                    0.067372555,
                    -0.041464937,
                    -0.016387336]), 11)
)

co3 = Wavelet(
    Filter(np.array([-3.46e-05,
		-7.10e-05,
		0.000466217,
		0.001117519,
		-0.002574518,
		-0.009007976,
		0.015880545,
		0.034555028,
		-0.082301927,
		-0.071799822,
		0.428483476,
		0.793777223,
		0.405176902,
		-0.06112339,
		-0.065771911,
		0.023452696,
		0.007782596,
		-0.003793513]), 17),
    Filter(np.array([0.003793513,
		0.007782596,
		-0.023452696,
		-0.065771911,
		0.06112339,
		0.405176902,
		-0.793777223,
		0.428483476,
		0.071799822,
		-0.082301927,
		-0.034555028,
		0.015880545,
		0.009007976,
		-0.002574518,
		-0.001117519,
		0.000466217,
		7.10e-05,
		-3.46e-05 ]), 0),
    Filter(np.array([-0.003793513,
		0.007782596,
		0.023452696,
		-0.065771911,
		-0.06112339,
		0.405176902,
		0.793777223,
		0.428483476,
		-0.071799822,
		-0.082301927,
		0.034555028,
		0.015880545,
		-0.009007976,
		-0.002574518,
		0.001117519,
		0.000466217,
		-7.10e-05,
		-3.46e-05]), 0),
    Filter(np.array([-3.46e-05,
		7.10e-05,
		0.000466217,
		-0.001117519,
		-0.002574518,
		0.009007976,
		0.015880545,
		-0.034555028,
		-0.082301927,
		0.071799822,
		0.428483476,
		-0.793777223,
		0.405176902,
		0.06112339,
		-0.065771911,
		-0.023452696,
		0.007782596,
		0.003793513]), 17)
)

def get_wavelet(wavelet_name):
    """
    Get a wavelet based on the wavelets name.

    Args:
        wavelet_name (str): Name of the wavelet ('haar', 'db1', 'db2', 'db3' or 'db4').

    Returns:
        A wavelet object. If the wavelet name is not recognized, it returns None.
    """
    wname = wavelet_name.lower()
    if wname == 'db1' or wname == 'haar':
        return db1
    elif wname == 'db2':
        return db2
    elif wname == 'db3':
        return db3
    elif wname == 'db4':
        return db4
    elif wname == 'db5':
        return db5
    elif wname == 'db6':
        return db6
    elif wname == 'db7':
        return db7
    elif wname == 'sy4':
        return sy4
    elif wname == 'sy5':
        return sy5
    elif wname == 'co1':
        return co1
    elif wname == 'co2':
        return co2
    elif wname == 'co3':
        return co3
    elif wname == 'sy8':
        return sy8
    else:
        return None


