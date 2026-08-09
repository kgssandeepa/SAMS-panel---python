from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from app.models.image_data import ImageData
from app.utils.exceptions import ImageProcessingError
from app.utils.logger import logger


class GridGenerator:
    
    # Generate table grid from detected intersections.

    DEFAULT_TOLERANCE = 10

    @staticmethod
    def generate(
        image_data: ImageData,
        tolerance: int = DEFAULT_TOLERANCE,
    ):
        
        # Main grid generation pipeline.

        logger.info(
            "Generating table grid..."
        )

        if image_data.intersections is None:

            raise ImageProcessingError(
                "No intersections detected."
            )

        rows = GridGenerator.group_rows(
            image_data.intersections,
            tolerance,
        )

        columns = GridGenerator.group_columns(
            image_data.intersections,
            tolerance,
        )

        rows = GridGenerator.sort_rows(rows)

        columns = GridGenerator.sort_columns(columns)

        image_data.grid_rows = rows

        image_data.grid_columns = columns

        image_data.processing_history[
            "Grid Generation"
        ] = {
            "rows": len(rows),
            "columns": len(columns),
        }

        image_data.set_stage(
            "Grid Generation"
        )

        logger.info(
            "Grid generated successfully."
        )

        return rows, columns

    @staticmethod
    def group_rows(
        points,
        tolerance,
    ):
        
        # Group points into rows.

        grouped = defaultdict(list)

        for x, y in sorted(
            points,
            key=lambda p: p[1],
        ):

            assigned = False

            for key in list(grouped.keys()):

                if abs(
                    key - y
                ) <= tolerance:

                    grouped[key].append(
                        (x, y)
                    )

                    assigned = True

                    break

            if not assigned:

                grouped[y].append(
                    (x, y)
                )

        return list(
            grouped.values()
        )

    @staticmethod
    def group_columns(
        points,
        tolerance,
    ):
        
        # Group points into columns.

        grouped = defaultdict(list)

        for x, y in sorted(
            points,
            key=lambda p: p[0],
        ):

            assigned = False

            for key in list(grouped.keys()):

                if abs(
                    key - x
                ) <= tolerance:

                    grouped[key].append(
                        (x, y)
                    )

                    assigned = True

                    break

            if not assigned:

                grouped[x].append(
                    (x, y)
                )

        return list(
            grouped.values()
        )

    @staticmethod
    def sort_rows(
        rows,
    ):
        
        # Sort every row.

        sorted_rows = []

        for row in rows:

            sorted_rows.append(

                sorted(
                    row,
                    key=lambda p: p[0],
                )

            )

        sorted_rows.sort(
            key=lambda r: r[0][1]
        )

        return sorted_rows

    @staticmethod
    def sort_columns(
        columns,
    ):
        
        # Sort every column.

        sorted_columns = []

        for column in columns:

            sorted_columns.append(

                sorted(
                    column,
                    key=lambda p: p[1],
                )

            )

        sorted_columns.sort(
            key=lambda c: c[0][0]
        )

        return sorted_columns
    
    @staticmethod
    def create_grid_image(
        image_data: ImageData,
    ) -> np.ndarray:
        
        # Draw the detected grid on a blank image.

        if (
            image_data.grid_rows is None
            or
            image_data.grid_columns is None
        ):
            raise ImageProcessingError(
                "Grid has not been generated."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        height, width = image.shape[:2]

        grid = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        # Draw horizontal lines

        for row in image_data.grid_rows:

            if len(row) < 2:
                continue

            for i in range(len(row) - 1):

                cv2.line(

                    grid,

                    row[i],

                    row[i + 1],

                    255,

                    2,

                )

        # Draw vertical lines

        for column in image_data.grid_columns:

            if len(column) < 2:
                continue

            for i in range(len(column) - 1):

                cv2.line(

                    grid,

                    column[i],

                    column[i + 1],

                    255,

                    2,

                )

        image_data.grid_image = grid

        return grid

    @staticmethod
    def extract_cells(
        image_data: ImageData,
    ):
        
        # Generate table cells from neighboring intersections.

        if image_data.grid_rows is None:

            raise ImageProcessingError(
                "Grid rows unavailable."
            )

        rows = image_data.grid_rows

        cells = []

        for r in range(len(rows) - 1):

            upper = rows[r]

            lower = rows[r + 1]

            limit = min(
                len(upper),
                len(lower),
            )

            for c in range(limit - 1):

                cell = GridGenerator.create_cell(

                    upper[c],

                    upper[c + 1],

                    lower[c],

                    lower[c + 1],

                    row=r,

                    column=c,

                )

                cells.append(cell)

        image_data.table_cells = cells

        logger.info(

            f"{len(cells)} table cells generated."

        )

        return cells

    @staticmethod
    def create_cell(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        row=0,
        column=0,
    ):
        
        # Create one cell dictionary.

        x1, y1 = top_left

        x2, _ = top_right

        _, y2 = bottom_left

        return {

            "top_left": top_left,

            "top_right": top_right,

            "bottom_left": bottom_left,

            "bottom_right": bottom_right,

            "row": row,

            "column": column,

            "bbox": (

                x1,

                y1,

                x2 - x1,

                y2 - y1,

            ),

        }

    @staticmethod
    def cell_count(
        image_data: ImageData,
    ) -> int:
        
        # Return number of cells.

        if image_data.table_cells is None:

            return 0

        return len(
            image_data.table_cells
        )

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:
        
        # Grid statistics.

        rows = (
            len(image_data.grid_rows)
            if image_data.grid_rows
            else 0
        )

        columns = (
            len(image_data.grid_columns)
            if image_data.grid_columns
            else 0
        )

        cells = (
            len(image_data.table_cells)
            if image_data.table_cells
            else 0
        )

        return {

            "Rows": rows,

            "Columns": columns,

            "Cells": cells,

            "Intersections": (
                len(image_data.intersections)
                if image_data.intersections
                else 0
            ),

        }
    
    @staticmethod
    def overlay(
        image_data: ImageData,
        color=(0, 255, 0),
        radius: int = 4,
        thickness: int = 2,
    ) -> np.ndarray:
        
        # Draw the generated grid and detected intersections on the original image.

        if image_data.image is None:
            raise ImageProcessingError(
                "Original image not available."
            )

        if image_data.grid_rows is None:
            raise ImageProcessingError(
                "Grid has not been generated."
            )

        image = image_data.perspective_image if image_data.perspective_image is not None else image_data.image
        overlay = image.copy()

        # Draw horizontal grid lines
        for row in image_data.grid_rows:

            if len(row) < 2:
                continue

            for i in range(len(row) - 1):

                cv2.line(
                    overlay,
                    row[i],
                    row[i + 1],
                    color,
                    thickness,
                )

        # Draw vertical grid lines
        for column in image_data.grid_columns:

            if len(column) < 2:
                continue

            for i in range(len(column) - 1):

                cv2.line(
                    overlay,
                    column[i],
                    column[i + 1],
                    color,
                    thickness,
                )

        # Draw intersection points
        if image_data.intersections is not None:

            for point in image_data.intersections:

                cv2.circle(
                    overlay,
                    point,
                    radius,
                    (0, 0, 255),
                    -1,
                )

        return overlay

    @staticmethod
    def preview(
        image_data: ImageData,
    ):
        
        # Return generated grid image.

        if image_data.grid_image is None:

            return GridGenerator.create_grid_image(
                image_data
            )

        return image_data.grid_image.copy()

    @staticmethod
    def validate(
        image_data: ImageData,
    ) -> bool:
        
        # Validate generated grid.

        if image_data.grid_rows is None:
            return False

        if image_data.grid_columns is None:
            return False

        if len(image_data.grid_rows) < 2:
            return False

        if len(image_data.grid_columns) < 2:
            return False

        return True

    @staticmethod
    def export_grid(
        image_data: ImageData,
    ) -> list[dict]:
        
        # Export grid information.

        if image_data.table_cells is None:

            return []

        export = []

        for index, cell in enumerate(
            image_data.table_cells,
            start=1,
        ):

            export.append(

                {
                    "id": index,
                    "bbox": cell["bbox"],
                    "top_left": cell["top_left"],
                    "top_right": cell["top_right"],
                    "bottom_left": cell["bottom_left"],
                    "bottom_right": cell["bottom_right"],
                }

            )

        return export

    @staticmethod
    def reset(
        image_data: ImageData,
    ) -> None:
        
        # Reset generated grid.

        image_data.grid_rows = None

        image_data.grid_columns = None

        image_data.grid_image = None

        image_data.table_cells = None

        image_data.processing_history.pop(
            "Grid Generation",
            None,
        )

        logger.info(
            "Grid generator reset."
        )