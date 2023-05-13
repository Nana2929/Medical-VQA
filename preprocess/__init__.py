import cv2
import numpy as np
from pathlib import Path

from .config import _DEFAULT_HU_TRANSFORM_PARAMS
from ._transform import hu_transform
from ._tilt import adjust_tilt


_DEFAULT_PIPELINE_STEPS = {
    'HEAD': [
        (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['HEAD']),
        (adjust_tilt, 'HEAD')
    ],
    'ABD': [
        (hu_transform, *_DEFAULT_HU_TRANSFORM_PARAMS['ABD']),
        (adjust_tilt, 'ABD')
    ],
    'CHEST': [
        (adjust_tilt, 'CHEST')
    ]
}


def pipeline(imgset, type,  steps=None, out_dir=None):
    results = []
    steps = steps or _DEFAULT_PIPELINE_STEPS[type]
    for img_name, img in imgset:
        result = img.copy()

        print(img_name, '\r\n')
        for i, step in enumerate(steps):
            method, *args = step
            print(method.__name__, '\r\n')
            result = method(np.uint8(result), *args)

            if out_dir:
                dir_path = f'{out_dir}/{type}/{i+1}_{method.__name__}'
                Path(dir_path).mkdir(parents=True, exist_ok=True)

                cv2.imwrite(f'{dir_path}/{img_name}', result)

        results.append(result)
        print('=' * 30)

    return results


__all__ = ['pipeline', 'hu_transform', 'adjust_tilt']