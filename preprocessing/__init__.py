import shutil
import cv2
import numpy as np
from pathlib import Path

from .config import _DEFAULT_HU_TRANSFORM_PARAMS, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, _CANNY_MIN, _CANNY_MAX
from ._removing import remove_text
from ._transformation import hu_transform
from ._correction import adjust_tilt


_DEFAULT_PIPELINE_STEPS = {
    'HEAD': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN, _CANNY_MAX)),
        (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['HEAD']),
        (adjust_tilt, 'HEAD')
    ],
    'ABD': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN, _CANNY_MAX)),
        (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['ABD']),
        # (adjust_tilt, 'ABD')
    ],
    'CHEST': [
        (remove_text, _MORPHOLOGY_KERNEL, _GAUSS_VALUE, (_CANNY_MIN, _CANNY_MAX)),
        # (adjust_tilt, 'CHEST')
    ]
}


def pipeline(imgset, type,  steps=None, out_dir=None):
    results = []
    steps = steps or _DEFAULT_PIPELINE_STEPS[type]

    if out_dir:
        out_dir_path = f'{out_dir}/{type}'
        if Path(out_dir_path).exists():
            shutil.rmtree(out_dir_path)

    for img_name, img in imgset:
        result = img.copy()

        print(img_name, '\r\n')
        for i, step in enumerate(steps):
            method, *args = step
            print(method.__name__, '\r\n')
            result = method(np.uint8(result), *args)

            if out_dir:
                out_path = f'{out_dir_path}/{i+1}_{method.__name__}'
                Path(out_path).mkdir(parents=True, exist_ok=True)

                cv2.imwrite(f'{out_path}/{img_name}', result)

        results.append(result)
        print('=' * 30)

    return results


__all__ = ['pipeline', 'remove_text', 'hu_transform', 'adjust_tilt']