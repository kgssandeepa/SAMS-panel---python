from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class Morphology:

    DEFAULT_KERNEL = (3, 3)

    @staticmethod
    def kernel(size=(3,3)):

        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            size,
        )

    @staticmethod
    def opening(
        image_data: ImageData,
        kernel_size=(3,3),
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        if len(source.shape) == 3 and source.shape[2] == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        result = cv2.morphologyEx(
            source,
            cv2.MORPH_OPEN,
            Morphology.kernel(kernel_size),
        )

        image_data.opened_image = result

        image_data.processing_history[
            "Opening"
        ] = result

        image_data.set_stage(
            "Morphological Opening"
        )

        return result

    @staticmethod
    def closing(
        image_data: ImageData,
        kernel_size=(3,3),
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        if len(source.shape) == 3 and source.shape[2] == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        result = cv2.morphologyEx(
            source,
            cv2.MORPH_CLOSE,
            Morphology.kernel(kernel_size),
        )

        image_data.closed_image = result

        image_data.processing_history[
            "Closing"
        ] = result

        image_data.set_stage(
            "Morphological Closing"
        )

        return result

    @staticmethod
    def dilate(
        image_data: ImageData,
        iterations=1,
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        return cv2.dilate(
            source,
            Morphology.kernel(),
            iterations=iterations,
        )

    @staticmethod
    def erode(
        image_data: ImageData,
        iterations=1,
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        return cv2.erode(
            source,
            Morphology.kernel(),
            iterations=iterations,
        )

    @staticmethod
    def gradient(
        image_data: ImageData,
    ):

        source = ImageSelector.get_latest(
            image_data
        )

        return cv2.morphologyEx(
            source,
            cv2.MORPH_GRADIENT,
            Morphology.kernel(),
        )

    @staticmethod
    def statistics(
        image_data: ImageData,
    ):

        if image_data.closed_image is not None:

            return ImageStatistics.basic(
                image_data.closed_image
            )

        if image_data.opened_image is not None:

            return ImageStatistics.basic(
                image_data.opened_image
            )

        raise ImageProcessingError(
            "Morphology image not available."
        )

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.opened_image = None

        image_data.closed_image = None

        image_data.processing_history.pop(
            "Opening",
            None,
        )

        image_data.processing_history.pop(
            "Closing",
            None,
        )

        logger.info(
            "Morphology reset."
        )