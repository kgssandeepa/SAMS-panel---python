from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.image_statistics import ImageStatistics
from app.utils.logger import logger


class TableDetector:
    
    # Detect attendance table.

    MIN_TABLE_AREA = 5000

    @staticmethod
    def detect(
        image_data: ImageData,
    ) -> np.ndarray:

        try:

            logger.info(
                "Detecting attendance table..."
            )

            if image_data.merged_lines is None:

                raise ImageProcessingError(
                    "Merged grid image not available."
                )

            image = image_data.merged_lines.copy()

            contours, _ = cv2.findContours(
                image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:

                raise ImageProcessingError(
                    "No table detected."
                )

            largest = max(
                contours,
                key=cv2.contourArea,
            )

            area = cv2.contourArea(
                largest
            )

            if area < TableDetector.MIN_TABLE_AREA:

                raise ImageProcessingError(
                    "Detected contour is too small."
                )

            image_data.table_contour = largest

            x, y, w, h = cv2.boundingRect(
                largest
            )

            image_data.table_bbox = (
                x,
                y,
                w,
                h,
            )

            mask = np.zeros_like(image)

            cv2.drawContours(
                mask,
                [largest],
                -1,
                255,
                thickness=-1,
            )

            image_data.table_mask = mask

            image_data.processing_history[
                "Table Detection"
            ] = mask

            image_data.set_stage(
                "Table Detection"
            )

            logger.info(
                "Attendance table detected."
            )

            return mask

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                str(ex)
            ) from ex

    @staticmethod
    def extract(
        image_data: ImageData,
    ) -> np.ndarray:

        if image_data.table_bbox is None:

            raise ImageProcessingError(
                "Table not detected."
            )

        x, y, w, h = image_data.table_bbox

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        return image[
            y:y+h,
            x:x+w,
        ]

    @staticmethod
    def overlay(
        image_data: ImageData,
        color=(0,255,0),
        thickness=3,
    ) -> np.ndarray:

        if image_data.table_contour is None:

            raise ImageProcessingError(
                "Table contour not found."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        overlay = image.copy()

        cv2.drawContours(
            overlay,
            [image_data.table_contour],
            -1,
            color,
            thickness,
        )

        return overlay

    @staticmethod
    def draw_bbox(
        image_data: ImageData,
        color=(255,0,0),
        thickness=2,
    ) -> np.ndarray:

        if image_data.table_bbox is None:

            raise ImageProcessingError(
                "Bounding box unavailable."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        overlay = image.copy()

        x,y,w,h = image_data.table_bbox

        cv2.rectangle(
            overlay,
            (x,y),
            (x+w,y+h),
            color,
            thickness,
        )

        return overlay

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:

        if image_data.table_contour is None:

            raise ImageProcessingError(
                "Table not detected."
            )

        stats = ImageStatistics.basic(
            image_data.table_mask
        )

        x,y,w,h = image_data.table_bbox

        stats.update({

            "Table Area":
                cv2.contourArea(
                    image_data.table_contour
                ),

            "Bounding Width":
                w,

            "Bounding Height":
                h,

            "Aspect Ratio":
                round(
                    w/h,
                    2,
                ),

            "Bounding Box":
                image_data.table_bbox,

        })

        return stats

    @staticmethod
    def contour_area(
        image_data: ImageData,
    ) -> float:

        if image_data.table_contour is None:

            return 0.0

        return float(

            cv2.contourArea(

                image_data.table_contour

            )

        )

    @staticmethod
    def contour_perimeter(
        image_data: ImageData,
    ) -> float:

        if image_data.table_contour is None:

            return 0.0

        return float(

            cv2.arcLength(

                image_data.table_contour,

                True,

            )

        )

    @staticmethod
    def preview(
        image_data: ImageData,
    ):

        return image_data.table_mask

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.table_contour = None

        image_data.table_mask = None

        image_data.table_bbox = None

        image_data.processing_history.pop(

            "Table Detection",

            None,

        )

        logger.info(

            "Table detector reset."

        )