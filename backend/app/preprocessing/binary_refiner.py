from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class BinaryRefiner:

    DEFAULT_KERNEL = (3, 3)

    @staticmethod
    def _kernel(size):

        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            size,
        )

    @staticmethod
    def refine(
        image_data: ImageData,
        kernel_size=(3,3),
    ):

        try:

            logger.info(
                "Refining binary image..."
            )

            source = ImageSelector.get_latest(
                image_data
            )

            if len(source.shape) == 3:
                source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

            _, source = cv2.threshold(source, 127, 255, cv2.THRESH_BINARY)

            kernel = BinaryRefiner._kernel(
                kernel_size
            )

            result = cv2.morphologyEx(
                source,
                cv2.MORPH_CLOSE,
                kernel,
            )

            result = cv2.morphologyEx(
                result,
                cv2.MORPH_OPEN,
                kernel,
            )

            image_data.binary_image = result

            image_data.processing_history[
                "Binary Refinement"
            ] = result

            image_data.set_stage(
                "Binary Refinement"
            )

            return result

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                str(ex)
            ) from ex

    @staticmethod
    def remove_noise(
        image_data: ImageData,
        kernel_size=(3,3),
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        return cv2.morphologyEx(
            source,
            cv2.MORPH_OPEN,
            BinaryRefiner._kernel(kernel_size),
        )

    @staticmethod
    def fill_holes(
        image_data: ImageData,
        kernel_size=(3,3),
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        return cv2.morphologyEx(
            source,
            cv2.MORPH_CLOSE,
            BinaryRefiner._kernel(kernel_size),
        )

    @staticmethod
    def statistics(
        image_data: ImageData,
    ):

        if image_data.binary_image is None:

            raise ImageProcessingError(
                "Binary image not available."
            )

        return ImageStatistics.basic(
            image_data.binary_image
        )

    @staticmethod
    def preview(
        image_data: ImageData,
    ):

        return image_data.binary_image

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.binary_image = None

        image_data.processing_history.pop(
            "Binary Refinement",
            None,
        )

        logger.info(
            "Binary refinement reset."
        )