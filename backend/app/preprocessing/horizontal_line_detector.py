from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_selector import ImageSelector
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class HorizontalLineDetector:
    
    # Detect horizontal table lines.

    DEFAULT_SCALE = 12
    MIN_SCALE = 5

    @staticmethod
    def detect(
        image_data: ImageData,
        scale: int = DEFAULT_SCALE,
    ) -> np.ndarray:
        
        # Detect horizontal lines from the latest binary image.

        try:

            logger.info(
                "Detecting horizontal lines..."
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
                HorizontalLineDetector.MIN_SCALE,
                scale,
            )

            kernel_width = min(120, max(40, width // 15))

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_width, 1),
            )

            horizontal = cv2.erode(
                source,
                kernel,
                iterations=1,
            )

            horizontal = cv2.dilate(
                horizontal,
                kernel,
                iterations=1,
            )

            image_data.horizontal_lines = horizontal

            image_data.processing_history[
                "Horizontal Line Detection"
            ] = horizontal

            image_data.set_stage(
                "Horizontal Line Detection"
            )

            logger.info(
                "Horizontal lines detected successfully."
            )

            return horizontal

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Horizontal line detection failed: {ex}"
            ) from ex

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:
        
        # Return image statistics.

        if image_data.horizontal_lines is None:

            raise ImageProcessingError(
                "Horizontal line image not available."
            )

        stats = ImageStatistics.basic(
            image_data.horizontal_lines
        )

        stats["Detected Pixels"] = int(
            np.count_nonzero(
                image_data.horizontal_lines
            )
        )

        return stats

    @staticmethod
    def preview(
        image_data: ImageData,
    ) -> np.ndarray:
        
        # Preview detected lines.

        if image_data.horizontal_lines is None:

            return HorizontalLineDetector.detect(
                image_data
            )

        return image_data.horizontal_lines.copy()

    @staticmethod
    def overlay(
        image_data: ImageData,
        color=(0, 255, 0),
    ) -> np.ndarray:
        
        # Overlay horizontal lines on the original image.

        if image_data.horizontal_lines is None:

            raise ImageProcessingError(
                "Horizontal lines not available."
            )

        if image_data.image is None:

            raise ImageProcessingError(
                "Original image not available."
            )

        overlay = image_data.image.copy()

        mask = image_data.horizontal_lines > 0

        overlay[mask] = color

        return overlay

    @staticmethod
    def count_lines(
        image_data: ImageData,
    ) -> int:
        
        # Estimate number of horizontal lines.

        if image_data.horizontal_lines is None:

            raise ImageProcessingError(
                "Horizontal lines not available."
            )

        contours, _ = cv2.findContours(
            image_data.horizontal_lines,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return len(contours)

    @staticmethod
    def reset(
        image_data: ImageData,
    ) -> None:
        
        # Reset results.

        image_data.horizontal_lines = None

        image_data.processing_history.pop(
            "Horizontal Line Detection",
            None,
        )

        logger.info(
            "Horizontal line detector reset."
        )