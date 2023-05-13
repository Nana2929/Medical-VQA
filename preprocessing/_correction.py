import abc
import math
import cv2


class TiltAdjuster:

    @abc.abstractmethod
    def _adjust(self, img):
        raise NotImplementedError

    def excute(self, img):
        tmp_img = img.copy()
        return self._adjust(tmp_img)


class HeadTiltAdjuster(TiltAdjuster):

    def _adjust(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        (x, y), (a, b), angle = cv2.fitEllipse(c)
        # cv2.ellipse(img, ((x,y), (a,b), angle), color=(0, 255, 0), thickness=2)

        rmajor = max(a, b) / 2

        if angle > 90:
            angle -= 90
        else:
            angle += 96

        # xtop = x + math.cos(math.radians(angle)) * rmajor
        # ytop = y + math.sin(math.radians(angle)) * rmajor
        # xbot = x + math.cos(math.radians(angle + 180)) * rmajor
        # ybot = y + math.sin(math.radians(angle + 180)) * rmajor
        # cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

        M = cv2.getRotationMatrix2D((x, y), angle - 90,
                                    1)  # transformation matrix
        corrected_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                       cv2.INTER_CUBIC)

        return corrected_img


class ChestTiltAdjuster(TiltAdjuster):

    def _adjust(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        (x, y), (a, b), angle = cv2.fitEllipse(c)
        # cv2.ellipse(img, ((x,y), (a,b), angle), color=(0, 255, 0), thickness=2)

        rmajor = max(a, b) / 2

        if angle > 90:
            angle -= 90
        else:
            angle += 96

        # xtop = x + math.cos(math.radians(angle)) * rmajor
        # ytop = y + math.sin(math.radians(angle)) * rmajor
        # xbot = x + math.cos(math.radians(angle + 180)) * rmajor
        # ybot = y + math.sin(math.radians(angle + 180)) * rmajor
        # cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

        M = cv2.getRotationMatrix2D((x, y), angle - 90,
                                    1)  # transformation matrix
        corrected_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                       cv2.INTER_CUBIC)

        return corrected_img


class AbdTiltAdjuster(TiltAdjuster):

    def _adjust(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        (x, y), (a, b), angle = cv2.fitEllipse(c)
        # cv2.ellipse(img, ((x,y), (a,b), angle), color=(0, 255, 0), thickness=2)

        rmajor = max(a, b) / 2

        if angle > 90:
            angle -= 90
        else:
            angle += 96

        # xtop = x + math.cos(math.radians(angle)) * rmajor
        # ytop = y + math.sin(math.radians(angle)) * rmajor
        # xbot = x + math.cos(math.radians(angle + 180)) * rmajor
        # ybot = y + math.sin(math.radians(angle + 180)) * rmajor
        # cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

        M = cv2.getRotationMatrix2D((x, y), angle - 90,
                                    1)  # transformation matrix
        corrected_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                       cv2.INTER_CUBIC)

        return corrected_img


def adjust_tilt(img, type):
    adjuster = None
    if type == 'HEAD':
        adjuster = HeadTiltAdjuster()
    elif type == 'CHEST':
        adjuster = ChestTiltAdjuster()
    elif type == 'ABD':
        adjuster = AbdTiltAdjuster()
    else:
        raise ValueError('Invalid type')

    corrected_img = adjuster.excute(img)

    return corrected_img