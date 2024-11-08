- full-pixel
- subpixel

Sensor noise is modeled as a Gaussian
random vector with uncorrelated components of equal variance.

First, it provides a
complete and self-contained theoretical derivation of a subpixel
target detector using the generalized likelihood ratio test (GLRT)
approach and the LMM

Second, it introduces
a systematic approach to investigate how well the adopted model
characterizes the data, and how robust the detection algorithm
is to model-data mismatches. Finally, it compares the derived
algorithms with regard to two desirable properties: capacity to
operate in constant false alarm rate (CFAR) mode and ability to
increase the separation between target and background.

HSI data processing attempts to accomplish from a distance
what a chemical spectroscopist does in the laboratory

### the following classes of application-specific tasks:

1. searching the pixels of an HSI data cube for “rare” pixels
with known spectral signatures (target detection), or for
pixels whose spectra significantly differ from the local
background (anomaly detection);
2. finding the “significant” (i.e., important to the user) spectral changes between two HSI scenes of the same geographic region (change detection);
3. assigning a label (class) to each pixel of a HSI data cube
(classification);
4. estimating the fraction of the pixel area covered by each
material present in the scene (unmixing).

the spectrum of
a subpixel target is mixed with the spectrum or spectra of the
background, the result is a pixel with a combined spectral signature. Hence, subpixel target detection should involve, either
directly or indirectly, some kind of (linear or nonlinear) separation of the constituent elements (unmixing).

### Linear Mixing Model

The basic premises of linear mixture modeling are that within
a given scene: a) the surface is dominated by a small number
of materials with relatively constant spectra (endmembers), b)
most of the spectral variability within the scene results from
varying proportions of the endmembers, and c) the mixing relationship is linear if the endmembers are arranged in spatially
distinct patterns [7].
In the linear mixing model (LMM), the spectrum of a mixed
pixel is represented [7] as a linear combination of component
spectra (endmembers). The weight of each endmember spectrum (abundance) is proportional to the fraction of the pixel area
covered by the endmember. If there are spectral bands, the
spectrum of the pixel and the spectra of the endmembers can be
represented by -dimensional vectors. Therefore, the general
equation for mixing by area is given by

![Untitled3](https://github.com/user-attachments/assets/08fcfd7e-fbfb-4375-80f9-ef7e8cf7559a)


Fitting a linear mixing model involves two steps: a) endmember identification and b) abundance estimation. Although
there are algorithms where the two steps are interwoven, the
objectives of this paper are better served by keeping the two
steps distinct. If we know the endmembers , unmixing can be
viewed either as a linear estimation problem or as a linear model
fitting problem. Furthermore, in detection and classification applications, we do not need to determine explicitly estimates of
; we can use a measure of the quality of the estimator or a measure of the model “goodness-of-fit” to make decisions. This is
the approach used in this paper.
