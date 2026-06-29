"""The location-interpretation axis (ADR-0011): which summary of a noise model's
distribution the deterministic prediction is taken to be.

PEtab v2 hardcodes the median; PyBNF makes it an explicit, overridable choice. It
only bites when the noise is asymmetric on the prediction's scale: on a symmetric
(e.g. linear-additive Gaussian) noise model mean = median = mode, so every
location coincides. A ``LocationInterpretation`` returns the additive-space offset
to subtract from ``scale.forward(prediction)`` to obtain the family's location
parameter.

The **median** commutes with the monotone scale transform, so its offset is 0 on
any scale -- which is why it is the clean, scale-agnostic default. The **mean**
picks up the family's moment correction in the transformed space -- which is
**family-specific** (Gaussian's differs from Laplace's, #419), so it is delegated
to the noise model's own ``mean_offset`` rather than computed here. ``MODE`` is not
modeled until a use exercises it.
"""


class LocationInterpretation:
    """The offset to subtract from ``scale.forward(prediction)`` to recover a
    family's additive-space location parameter -- 0 for the median, the family's
    moment correction for the mean."""

    def offset(self, noise_model, noise):
        raise NotImplementedError

    def d_offset_d_noise(self, noise_model, noise):
        """``d(offset)/d(noise parameter)`` -- the offset's own dependence on the noise scale,
        which the estimated-scale gradient column needs (#385). 0 for the median (its offset is
        identically 0); the family's moment-correction derivative for the mean."""
        raise NotImplementedError


class _Median(LocationInterpretation):
    def offset(self, noise_model, noise):
        return 0.0

    def d_offset_d_noise(self, noise_model, noise):
        return 0.0


class _Mean(LocationInterpretation):
    def offset(self, noise_model, noise):
        # The moment correction is family-specific (#419): ask the family.
        return noise_model.mean_offset(noise)

    def d_offset_d_noise(self, noise_model, noise):
        # The mean offset depends on the noise scale on a log scale (Gaussian's ln(base)sigma^2/2,
        # Laplace's -ln(1-b^2 t^2)/t); its derivative is family-specific, so ask the family. 0 on a
        # linear scale (the offset is 0), so the estimated-scale column reduces byte-for-byte (#385).
        return noise_model.d_mean_offset_d_noise(noise)


MEDIAN = _Median()
MEAN = _Mean()
