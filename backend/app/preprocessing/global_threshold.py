from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class GlobalThreshold:
    
    # Fixed global threshold.

    DEFAULT_THRESHOLD = 127
    MAX_VALUE = 255

    @staticmethod
    def apply(
        image_data: ImageData,
        threshold: int = DEFAULT_THRESHOLD,
        inverse: bool = False,
    ) -> np.ndarray:

        try:

            logger.info("Applying Global Threshold...")

            source = ImageSelector.get_latest(
                image_data
            )

            threshold_type = cv2.THRESH_BINARY

            if inverse:
                threshold_type = cv2.THRESH_BINARY_INV

            _, binary = cv2.threshold(
                source,
                threshold,
                GlobalThreshold.MAX_VALUE,
                threshold_type,
            )

            image_data.global_threshold_image = binary

            image_data.processing_history[
                "Global Threshold"
            ] = binary

            image_data.set_stage(
                "Global Threshold"
            )

            logger.info(
                "Global threshold completed."
            )

            return binary

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Global threshold failed: {ex}"
            ) from ex

    @staticmethod
    def statistics(
        image_data: ImageData,
    ):

        if image_data.global_threshold_image is None:

            raise ImageProcessingError(
                "Global threshold image missing."
            )

        return ImageStatistics.basic(
            image_data.global_threshold_image
        )

    @staticmethod
    def preview(
        image_data: ImageData,
    ):

        if image_data.global_threshold_image is None:

            return GlobalThreshold.apply(
                image_data
            )

        return image_data.global_threshold_image.copy()

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.global_threshold_image = None

        image_data.processing_history.pop(
            "Global Threshold",
            None,
        )

        logger.info(
            "Global threshold reset."
        )