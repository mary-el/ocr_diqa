from src.data.features.utils.libsmop import cell, copy, size, mod, concat, arange
import numpy as np
from numpy.fft import ifftshift
from src.data.features.utils.lowpassfilter import lowpassfilter


def logGabor_2D(im, nornt, ns, scalefac, sigma):
    # Gaussian describing the log Gabor filter's
    # transfer function in the frequency domain
    # to the filter center frequency.
    nscale = copy(ns)
    norient = copy(nornt)
    minWaveLength = 4
    r = copy(scalefac)
    sigmaOnf = copy(sigma)
    eps = 1e-9

    # Ratio of angular interval between filter orientations
    # and the standard deviation of the angular Gaussian
    # function used to construct filters in the
    # freq. plane.
    dThetaOnSigma = 1.5

    # Calculate the standard deviation of the
    # angular Gaussian function used to
    # construct filters in the freq. plane.
    thetaSigma = np.pi / norient / dThetaOnSigma

    rows, cols = size(im, nargout=2)

    # Pre-compute some stuff to speed up filter construction
    # Set up X and Y matrices with ranges normalised to +/- 0.5
    # The following code adjusts things appropriately for odd and even values
    # of rows and columns.
    if mod(cols, 2):
        xrange = concat([arange(-(cols - 1) / 2, (cols - 1) / 2)]) / (cols - 1)
    else:
        xrange = concat([arange(-cols / 2, (cols / 2 - 1))]) / cols

    if mod(rows, 2):
        yrange = concat([arange(-(rows - 1) / 2, (rows - 1) / 2)]) / (rows - 1)
    else:
        yrange = concat([arange(-rows / 2, (rows / 2 - 1))]) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(
        x**2 + y**2
    )  # Matrix values contain *normalised* radius from centre.
    # Matrix values contain polar angle.
    # (note -ve y is used to give +ve
    # anti-clockwise angles)
    theta = np.arctan2(-y, x)

    # Quadrant shift radius and theta so that filters
    # are constructed with 0 frequency at the corners.
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    # Get rid of the 0 radius value at the 0
    # frequency point (now at top-left corner)
    # so that taking the log of the radius will
    # not cause trouble.
    radius[0, 0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter
    #    responds to
    # 2) The angular component, which controls the orientation that the filter
    #    responds to.
    # The two components are multiplied together to construct the overall filter.

    # Construct the radial filter components...

    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    # calculating phase congrunecy.
    lp = lowpassfilter([rows, cols], 0.45, 15)  # Radius 0.45, sharpness 15

    logGabor = np.zeros((nscale.get(0), *radius.shape))

    # crr = 1;
    for s in range(nscale.get(0)):
        if size(r, 2) > 1:
            #         crr = crr * r(s);
            wavelength = minWaveLength * r[s]
        else:
            wavelength = minWaveLength * (r ** (s - 1))
        # Centre frequency of filter.
        fo = 1.0 / wavelength
        logGabor[s] = np.exp(
            (-((np.log(radius / fo + eps)) ** 2)) / (2 * np.log(sigmaOnf) ** 2)
        )
        # Apply low-pass filter
        logGabor[s] = np.multiply(logGabor[s], lp)
        # Set the value at the 0 frequency point of the filter
        # back to zero (undo the radius fudge).
        logGabor[s][0, 0] = 0

    # Then construct the angular filter components...

    spread = np.zeros((norient.get(0), *radius.shape))
    for o in range(1, norient.get(0) + 1):
        # Filter angle
        angl = ((o - 1) * np.pi) / norient
        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        # Difference in sine
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        # Difference in cosine
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        # Absolute angular distance
        dtheta = abs(np.arctan2(ds, dc))
        # Calculate the angular filter component
        spread[o - 1] = np.exp((-(dtheta**2)) / (2 * thetaSigma**2))

    gabor_filter = np.zeros((nscale.get(0), norient.get(0), *radius.shape))
    for o in range(norient.get(0)):
        # Filter angle.
        #   angl = (o-1)*pi/norient;
        for s in range(nscale.get(0)):
            gabor_filter[s, o] = np.multiply(logGabor[s], spread[o])

    return gabor_filter
