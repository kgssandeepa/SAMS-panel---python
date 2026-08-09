import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preprocessing.image_loader import ImageLoader

from app.services.table_service import TableService

from app.visualization.table_visualizer import (
    TableVisualizer,
)

from app.visualization.grid_visualizer import (
    GridVisualizer,
)


TEST_IMAGE = Path(
    "data/images/1.jpeg"
)


def main():

    print("=" * 60)

    print("TABLE DETECTION TEST")

    print("=" * 60)

    image_data = ImageLoader.load(
        TEST_IMAGE
    )

    print()

    print("Image Loaded Successfully")

    print(image_data.summary())

    print()

    print("Running Table Detection...")

    TableService.process(
        image_data
    )

    print("Completed Successfully")

    print()

    stats = TableService.statistics(
        image_data
    )

    print("=" * 60)

    print("PIPELINE STATISTICS")

    print("=" * 60)

    for key, value in stats.items():

        print(f"{key:20}: {value}")

    print()

    print("=" * 60)

    print("GRID INFORMATION")

    print("=" * 60)

    grid_info = GridVisualizer.information(
        image_data
    )

    for key, value in grid_info.items():

        print(f"{key:20}: {value}")

    print()

    print("=" * 60)

    print("VISUALIZATIONS")

    print("=" * 60)

    TableVisualizer.comparison(
        image_data
    )

    GridVisualizer.comparison(
        image_data
    )

    print()

    print("Saving visualization results...")

    TableVisualizer.save_all(

        image_data,

        "results/table_detection",

    )

    GridVisualizer.save_all(

        image_data,

        "results/table_detection",

    )

    print("Saved Successfully")

    print()

    exported = TableService.export(
        image_data
    )

    print("=" * 60)

    print("EXPORTED CELLS")

    print("=" * 60)

    print(f"Total Cells : {len(exported)}")

    print()

    for cell in exported[:10]:

        print(cell)

    print()

    print("=" * 60)

    print("VALIDATION")

    print("=" * 60)

    print(

        "Grid Valid :",

        TableService.validate(
            image_data
        ),

    )

    print()

    print("TEST PASSED")

    print("=" * 60)


if __name__ == "__main__":

    main()