from __future__ import annotations

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.logger import logger


class IntersectionDetector:
    
    # Detect intersections of table grid lines.

    @staticmethod
    def detect(
        image_data: ImageData,
        min_distance: int = 8,
    ) -> list[tuple[int, int]]:

        try:

            logger.info(
                "Detecting table intersections..."
            )

            if image_data.horizontal_lines is None:

                raise ImageProcessingError(
                    "Horizontal lines not detected."
                )

            if image_data.vertical_lines is None:

                raise ImageProcessingError(
                    "Vertical lines not detected."
                )

            intersections = cv2.bitwise_and(
                image_data.horizontal_lines,
                image_data.vertical_lines,
            )

            contours, _ = cv2.findContours(
                intersections,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            points = []

            for contour in contours:

                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cx = int(
                    M["m10"] / M["m00"]
                )

                cy = int(
                    M["m01"] / M["m00"]
                )

                duplicate = False

                for px, py in points:

                    if (
                        abs(cx - px) < min_distance
                        and
                        abs(cy - py) < min_distance
                    ):
                        duplicate = True
                        break

                if not duplicate:

                    points.append((cx, cy))

            points = sorted(
                points,
                key=lambda p: (
                    p[1],
                    p[0],
                ),
            )

            image_data.intersections = points

            image_data.processing_history[
                "Intersection Detection"
            ] = points

            image_data.set_stage(
                "Intersection Detection"
            )

            logger.info(
                f"{len(points)} intersections detected."
            )

            return points

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                str(ex)
            ) from ex

    @staticmethod
    def overlay(
        image_data: ImageData,
        radius: int = 4,
        color=(0,0,255),
    ) -> np.ndarray:

        if image_data.intersections is None:

            raise ImageProcessingError(
                "No intersections detected."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        overlay = image.copy()

        for x, y in image_data.intersections:

            cv2.circle(
                overlay,
                (x, y),
                radius,
                color,
                -1,
            )

        return overlay

    @staticmethod
    def count(
        image_data: ImageData,
    ) -> int:

        if image_data.intersections is None:

            return 0

        return len(
            image_data.intersections
        )

    @staticmethod
    def rows(
        image_data: ImageData,
        tolerance: int = 10,
    ) -> list[list[tuple[int,int]]]:

        if image_data.intersections is None:

            raise ImageProcessingError(
                "No intersections."
            )

        rows = []

        for point in image_data.intersections:

            x, y = point

            added = False

            for row in rows:

                if abs(
                    row[0][1] - y
                ) <= tolerance:

                    row.append(point)

                    added = True

                    break

            if not added:

                rows.append([point])

        for row in rows:

            row.sort(
                key=lambda p: p[0]
            )

        return rows

    @staticmethod
    def columns(
        image_data: ImageData,
        tolerance: int = 10,
    ) -> list[list[tuple[int,int]]]:

        if image_data.intersections is None:

            raise ImageProcessingError(
                "No intersections."
            )

        columns = []

        points = sorted(
            image_data.intersections,
            key=lambda p: (
                p[0],
                p[1],
            ),
        )

        for point in points:

            x, y = point

            added = False

            for column in columns:

                if abs(
                    column[0][0] - x
                ) <= tolerance:

                    column.append(point)

                    added = True

                    break

            if not added:

                columns.append([point])

        for column in columns:

            column.sort(
                key=lambda p: p[1]
            )

        return columns

    @staticmethod
    def preview(
        image_data: ImageData,
    ):

        return image_data.intersections

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        image_data.intersections = None

        image_data.processing_history.pop(
            "Intersection Detection",
            None,
        )

        logger.info(
            "Intersection detector reset."
        )