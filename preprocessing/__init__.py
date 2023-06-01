import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .config import (_DEFAULT_HU_TRANSFORM_PARAMS, _MORPHOLOGY_KERNEL,
                     _GAUSS_VALUE, _CANNY_MIN, _CANNY_MAX, _FCM_NORM_VALUE,
                     _TO_APPLY_ABD_FILTERS, _CT_NOISE_FUNCS,
                     _SKIP_TILT_CORRECTION)
from ._text_removing import remove_text
from ._transformation import hu_transform
from ._correction import adjust_tilt
from ._head_mri_preproc import fcm_norm
from ._xray_preproc import norm, equal_hist, create_clahe, gauss_blur
from ._ct_preproc import wiener_filter, median_filter

_DEFAULT_PIPELINE_STEPS = {
    'HEAD_CT':
    [(remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN, _CANNY_MAX)),
     (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['HEAD']),
     (adjust_tilt, 'HEAD')],
    'HEAD_MRI': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN,
                                                         _CANNY_MAX)),
        # (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['HEAD']),
        (adjust_tilt, 'HEAD'),
        (fcm_norm, _FCM_NORM_VALUE),
    ],
    'ABD_CT': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN,
                                                         _CANNY_MAX)),
        # (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['ABD']),
        (median_filter, ),
        # (wiener_filter, ),
        # (adjust_tilt, 'ABD')
    ],
    'CHEST_X-Ray': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN,
                                                         _CANNY_MAX)),
        (norm, ),
        (create_clahe, 1, 16),
        (gauss_blur, ),
        # (adjust_tilt, 'CHEST')
    ]
}


def pipeline(imgset, type: str, steps=None, out_dir=None):
    """image preprocessing pipeline (not question-specific)
    Parameters
    ----------
    imgset : _type_
        the image set to be processed
    type : str
        target_modality (in the form: {mod}_{organ})
    steps : List[Tuple], optional
        preprocessing steps (functions) to be executed, by default None
    out_dir : str, optional
       output directory of the preprocessed images, by default None
    """
    results = []
    steps = steps or _DEFAULT_PIPELINE_STEPS[type]

    if out_dir:
        out_dir_path = f'{out_dir}/{type}'
        if Path(out_dir_path).exists():
            raise FileExistsError(
                f'Output directory {out_dir_path} already exists.')

    for img_name, img in tqdm(imgset):
        result = img.copy()

        for i, step in enumerate(steps):
            method, *args = step
            if (type == 'ABD_CT' and method.__name__ in _CT_NOISE_FUNCS
                ) and img_name not in _TO_APPLY_ABD_FILTERS:
                ...
            elif method.__name__ == 'adjust_tilt' and img_name in _SKIP_TILT_CORRECTION:
                ...
            else:
                try:
                    result = method(np.uint8(result), *args)
                except Exception as e:
                    print(
                        f'\tSkipped. Failed at {method.__name__} with {e}: the contour has no enough (5) points.'
                    )

            if out_dir:
                out_path = f'{out_dir_path}/{i+1}_{method.__name__}'

                Path(out_path).mkdir(parents=True, exist_ok=True)

                cv2.imwrite(f'{out_path}/{img_name}', result)

        results.append(result)

        # final writeout
        if out_dir:
            Path(f'{out_dir_path}/final').mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f'{out_dir_path}/final/{img_name}', result)

    return results


__all__ = ['pipeline']