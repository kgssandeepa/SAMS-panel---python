from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class OtsuThreshold:
    
    # Otsu automatic thresholding.

    MAX_VALUE = 255

    @staticmethod
    def apply(
        image_data: ImageData,
        inverse: bool = False,
    ) -> np.ndarray:

        try:

            logger.info(
                "Applying Otsu Threshold..."
            )

            source = ImageSelector.get_latest(
                image_data
            )

            if len(source.shape) == 3:
                source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

            threshold_type = (
                cv2.THRESH_BINARY |
                cv2.THRESH_OTSU
            )

            if inverse:

                threshold_type = (
                    cv2.THRESH_BINARY_INV |
                    cv2.THRESH_OTSU
                )

            threshold, binary = cv2.threshold(
                source,
                0,
                OtsuThreshold.MAX_VALUE,
                threshold_type,
            )

            image_data.otsu_image = binary

            image_data.otsu_threshold_value = float(
                threshold
            )

            image_data.processing_history[
                "Otsu Threshold"
            ] = binary

            image_data.set_stage(
                "Otsu Threshold"
            )

            logger.info(
                f"Otsu threshold = {threshold:.2f}"
            )

            return binary

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Otsu threshold failed: {ex}"
            ) from ex

    @staticmethod
    def threshold_value(
        image_data: ImageData,
    ) -> float:

        if hasattr(
            image_data,
            "otsu_threshold_value",
        ):

            return image_data.otsu_threshold_value

        raise ImageProcessingError(
            "Threshold not calculated."
        )

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:

        if image_data.otsu_image is None:

            raise ImageProcessingError(
                "Otsu image missing."
            )

        stats = ImageStatistics.basic(
            image_data.otsu_image
        )

        stats["Threshold"] = (
            image_data.otsu_threshold_value
        )

        return stats

    @staticmethod
    def preview(
        image_data: ImageData,
    ) -> np.ndarray:

        if image_data.otsu_image is None:

            return OtsuThreshold.apply(
                image_data
            )

        return image_data.otsu_image.copy()

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.otsu_image = None

        image_data.otsu_threshold_value = None

        image_data.processing_history.pop(
            "Otsu Threshold",
            None,
        )

        logger.info(
            "Otsu threshold reset."
        )