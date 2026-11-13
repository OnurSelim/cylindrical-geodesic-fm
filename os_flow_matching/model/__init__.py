import torch

from .vector_field import ConditionalVectorField

from .cylindrical_geodesic_unet import CylindricalGeodesicUNet
from .hilbert_naive_unet import HilbertNaiveUNet
from .ts_affine_unet import TimeSeriesUNet


def infer_model_kwargs_from_specs(
	model_class: type,
	source_spec: dict,
	condition_on_x0: bool = False,
) -> dict:
	n_src = sum(len(spec.get("src_mod", ())) for spec in (source_spec or {}).values())
	n_cond = sum(len(spec.get("cnd_mod", ())) for spec in (source_spec or {}).values())
	n_x0 = n_src if condition_on_x0 else 0

	if model_class is TimeSeriesUNet:
		return {
			"in_channels": n_src + n_x0 + n_cond,
			"out_channels": n_src,
		}

	if model_class is HilbertNaiveUNet:
		return {
			"in_channels": 2 * (n_src + n_x0 + n_cond),
			"out_channels": 2 * n_src,
		}

	if model_class is CylindricalGeodesicUNet:
		return {
			"in_channels": 3 * n_src + 3 * n_x0 + 3 * n_cond,
			"out_channels": 2,
		}

	return {}
