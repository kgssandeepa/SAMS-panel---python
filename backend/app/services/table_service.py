from __future__ import annotations

from app.models.image_data import ImageData

from app.preprocessing.horizontal_line_detector import (
    HorizontalLineDetector,
)

from app.preprocessing.vertical_line_detector import (
    VerticalLineDetector,
)

from app.preprocessing.line_merger import (
    LineMerger,
)

from app.preprocessing.table_detector import (
    TableDetector,
)

from app.preprocessing.intersection_detector import (
    IntersectionDetector,
)

from app.preprocessing.grid_generator import (
    GridGenerator,
)

from app.utils.exceptions import (
    ImageProcessingError,
)

from app.utils.logger import logger


class TableService:
    
    # Complete table detection pipeline.

    @staticmethod
    def process(
        image_data: ImageData,
    ) -> ImageData:
        
        # Execute complete table detection.

        logger.info(
            "Starting table detection pipeline..."
        )

        HorizontalLineDetector.detect(
            image_data
        )

        VerticalLineDetector.detect(
            image_data
        )

        LineMerger.merge(
            image_data
        )

        TableDetector.detect(
            image_data
        )

        IntersectionDetector.detect(
            image_data
        )

        GridGenerator.generate(
            image_data
        )

        GridGenerator.create_grid_image(
            image_data
        )

        GridGenerator.extract_cells(
            image_data
        )

        logger.info(
            "Table detection pipeline completed."
        )

        return image_data

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:
        
        # Return complete statistics.

        return {

            "Horizontal Lines":

                HorizontalLineDetector.count_lines(
                    image_data
                ),

            "Vertical Lines":

                VerticalLineDetector.count_lines(
                    image_data
                ),

            "Intersections":

                IntersectionDetector.count(
                    image_data
                ),

            "Rows":

                len(image_data.grid_rows)
                if image_data.grid_rows
                else 0,

            "Columns":

                len(image_data.grid_columns)
                if image_data.grid_columns
                else 0,

            "Cells":

                GridGenerator.cell_count(
                    image_data
                ),

            "Bounding Box":

                image_data.table_bbox,

        }

    @staticmethod
    def overlay(
        image_data: ImageData,
    ):
        
        # Return grid overlay.

        return GridGenerator.overlay(
            image_data
        )

    @staticmethod
    def export(
        image_data: ImageData,
    ):
        
        # Export detected cells.

        return GridGenerator.export_grid(
            image_data
        )

    @staticmethod
    def validate(
        image_data: ImageData,
    ) -> bool:
        
        # Validate generated grid.

        return GridGenerator.validate(
            image_data
        )

    @staticmethod
    def reset(
        image_data: ImageData,
    ) -> None:
        
        # Reset Previous results.

        HorizontalLineDetector.reset(
            image_data
        )

        VerticalLineDetector.reset(
            image_data
        )

        LineMerger.reset(
            image_data
        )

        TableDetector.reset(
            image_data
        )

        IntersectionDetector.reset(
            image_data
        )

        GridGenerator.reset(
            image_data
        )

        logger.info(
            "Table service reset completed."
        )

    @staticmethod
    def run(
        image_data: ImageData,
    ) -> ImageData:
        
        # Safe execution wrapper.

        try:

            return TableService.process(
                image_data
            )

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                f"Table detection failed: {ex}"
            ) from ex