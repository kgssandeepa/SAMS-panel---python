from __future__ import annotations

import matplotlib.pyplot as plt

from app.models.image_data import ImageData


class ThresholdVisualizer:
    
    # Visualization utilities for thresholding.

    @staticmethod
    def _show(
        image,
        title: str,
    ) -> None:

        if image is None:
            return

        plt.figure(figsize=(7, 7))

        plt.imshow(
            image,
            cmap="gray",
        )

        plt.title(title)

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def global_threshold(
        image_data: ImageData,
    ):

        ThresholdVisualizer._show(
            image_data.global_threshold_image,
            "Global Threshold",
        )

    @staticmethod
    def otsu(
        image_data: ImageData,
    ):

        ThresholdVisualizer._show(
            image_data.otsu_image,
            "Otsu Threshold",
        )

    @staticmethod
    def adaptive_mean(
        image_data: ImageData,
    ):

        ThresholdVisualizer._show(
            image_data.adaptive_mean_image,
            "Adaptive Mean Threshold",
        )

    @staticmethod
    def adaptive_gaussian(
        image_data: ImageData,
    ):

        ThresholdVisualizer._show(
            image_data.adaptive_gaussian_image,
            "Adaptive Gaussian Threshold",
        )

    @staticmethod
    def binary(
        image_data: ImageData,
    ):

        ThresholdVisualizer._show(
            image_data.binary_image,
            "Final Binary Image",
        )

    @staticmethod
    def pipeline(
        image_data: ImageData,
    ):

        images = [

            (
                "Global",
                image_data.global_threshold_image,
            ),

            (
                "Otsu",
                image_data.otsu_image,
            ),

            (
                "Adaptive Mean",
                image_data.adaptive_mean_image,
            ),

            (
                "Adaptive Gaussian",
                image_data.adaptive_gaussian_image,
            ),

            (
                "Binary",
                image_data.binary_image,
            ),
        ]

        valid = [

            (title, image)

            for title, image in images

            if image is not None

        ]

        if not valid:

            return

        columns = 3

        rows = (len(valid) + columns - 1) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(15, 5 * rows),
        )

        if rows == 1:

            axes = [axes] if columns == 1 else axes

        axes = (
            axes.flatten()
            if hasattr(axes, "flatten")
            else axes
        )

        for ax in axes:

            ax.axis("off")

        for ax, (title, image) in zip(
            axes,
            valid,
        ):

            ax.imshow(
                image,
                cmap="gray",
            )

            ax.set_title(title)

            ax.axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def summary(
        image_data: ImageData,
    ) -> None:

        print()

        print("=" * 60)

        print("Threshold Processing Summary")

        print("=" * 60)

        for stage in (

            "Global Threshold",

            "Otsu Threshold",

            "Adaptive Mean",

            "Adaptive Gaussian",

            "Binary Refinement",

        ):

            if stage in image_data.processing_history:

                print(f"✓ {stage}")

        print("=" * 60)