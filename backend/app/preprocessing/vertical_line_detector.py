from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class VerticalLineDetector:
    
    # Detect vertical table lines.

    DEFAULT_SCALE = 6
    MIN_SCALE = 3

    @staticmethod
    def detect(
        image_data: ImageData,
        scale: int = DEFAULT_SCALE,
    ) -> np.ndarray:
        
        # Detect vertical lines from the latest binary image.

        try:

            logger.info(
                "Detecting vertical lines..."
            )

            source = ImageSelector.get_latest(
                image_data
            )

            if source is None:

                raise ImageProcessingError(
                    "No binary image available."
                )

            if len(source.shape) == 3:

                source = cv2.cvtColor(
                    source,
                    cv2.COLOR_BGR2GRAY,
                )

            height, width = source.shape

            scale = max(
                VerticalLineDetector.MIN_SCALE,
                scale,
            )

            kernel_height = min(60, max(25, height // 15))

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, kernel_height),
            )

            vertical = cv2.erode(
                source,
                kernel,
                iterations=1,
            )

            vertical = cv2.dilate(
                vertical,
                kernel,
                iterations=1,
            )

            image_data.vertical_lines = vertical

            image_data.processing_history[
                "Vertical Line Detection"
            ] = vertical

            image_data.set_stage(
                "Vertical Line Detection"
            )

            logger.info(
                "Vertical lines detected successfully."
            )

            return vertical

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Vertical line detection failed: {ex}"
            ) from ex

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:
        
        # Return statistics for detected vertical lines.

        if image_data.vertical_lines is None:

            raise ImageProcessingError(
                "Vertical line image not available."
            )

        stats = ImageStatistics.basic(
            image_data.vertical_lines
        )

        stats["Detected Pixels"] = int(
            np.count_nonzero(
                image_data.vertical_lines
            )
        )

        return stats

    @staticmethod
    def preview(
        image_data: ImageData,
    ) -> np.ndarray:
        
        # Preview detected vertical lines.

        if image_data.vertical_lines is None:

            return VerticalLineDetector.detect(
                image_data
            )

        return image_data.vertical_lines.copy()

    @staticmethod
    def overlay(
        image_data: ImageData,
        color=(255, 0, 0),
    ) -> np.ndarray:
        
        # Overlay vertical lines on the original image.

        if image_data.vertical_lines is None:

            raise ImageProcessingError(
                "Vertical lines not available."
            )

        if image_data.image is None:

            raise ImageProcessingError(
                "Original image not available."
            )

        overlay = image_data.image.copy()

        mask = image_data.vertical_lines > 0

        overlay[mask] = color

        return overlay

    @staticmethod
    def count_lines(
        image_data: ImageData,
    ) -> int:
        
        # Estimate number of vertical lines.

        if image_data.vertical_lines is None:

            raise ImageProcessingError(
                "Vertical lines not available."
            )

        contours, _ = cv2.findContours(
            image_data.vertical_lines,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return len(contours)

    @staticmethod
    def reset(
        image_data: ImageData,
    ) -> None:
        
        # Reset detector results.

        image_data.vertical_lines = None

        image_data.processing_history.pop(
            "Vertical Line Detection",
            None,
        )

        logger.info(
            "Vertical line detector reset."
        )