from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class LineMerger:
    
    # Merge detected horizontal and vertical lines.

    DEFAULT_KERNEL = (3, 3)

    @staticmethod
    def merge(
        image_data: ImageData,
        kernel_size: tuple[int, int] = DEFAULT_KERNEL,
    ) -> np.ndarray:
        
        # Merge horizontal and vertical lines into one grid image.

        try:

            logger.info(
                "Merging horizontal and vertical lines..."
            )

            horizontal = image_data.horizontal_lines
            vertical = image_data.vertical_lines

            if horizontal is None:

                raise ImageProcessingError(
                    "Horizontal lines not found."
                )

            if vertical is None:

                raise ImageProcessingError(
                    "Vertical lines not found."
                )

            merged = cv2.bitwise_or(
                horizontal,
                vertical,
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                kernel_size,
            )

            merged = cv2.morphologyEx(
                merged,
                cv2.MORPH_CLOSE,
                kernel,
            )

            image_data.merged_lines = merged

            image_data.processing_history[
                "Line Merger"
            ] = merged

            image_data.set_stage(
                "Line Merger"
            )

            logger.info(
                "Line merging completed."
            )

            return merged

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Line merging failed: {ex}"
            ) from ex

    @staticmethod
    def repair(
        image_data: ImageData,
        kernel_size: tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        
        # Repair broken grid lines.

        if image_data.merged_lines is None:

            raise ImageProcessingError(
                "Merged image not available."
            )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size,
        )

        repaired = cv2.morphologyEx(
            image_data.merged_lines,
            cv2.MORPH_CLOSE,
            kernel,
        )

        image_data.merged_lines = repaired

        return repaired

    @staticmethod
    def overlay(
        image_data: ImageData,
        color=(0, 255, 0),
    ) -> np.ndarray:
        
        # Overlay merged grid on original image.

        if image_data.image is None:

            raise ImageProcessingError(
                "Original image not available."
            )

        if image_data.merged_lines is None:

            raise ImageProcessingError(
                "Merged grid not available."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        overlay = image.copy()

        mask = image_data.merged_lines > 0

        overlay[mask] = color

        return overlay

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:
        
        # Return merged grid statistics.

        if image_data.merged_lines is None:

            raise ImageProcessingError(
                "Merged grid not available."
            )

        stats = ImageStatistics.basic(
            image_data.merged_lines
        )

        stats["Grid Pixels"] = int(
            np.count_nonzero(
                image_data.merged_lines
            )
        )

        return stats

    @staticmethod
    def preview(
        image_data: ImageData,
    ) -> np.ndarray:
        
        # Preview merged grid.

        if image_data.merged_lines is None:

            return LineMerger.merge(
                image_data
            )

        return image_data.merged_lines.copy()

    @staticmethod
    def reset(
        image_data: ImageData,
    ) -> None:
        
        # Reset merger.

        image_data.merged_lines = None

        image_data.processing_history.pop(
            "Line Merger",
            None,
        )

        logger.info(
            "Line merger reset."
        )