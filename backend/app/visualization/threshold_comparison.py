from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from app.models.image_data import ImageData
from app.preprocessing.threshold_validator import ThresholdValidator


class ThresholdComparison:
    
    # Compare thresholding methods.

    @staticmethod
    def compare(
        image_data: ImageData,
    ) -> None:
        
        # Compare all threshold methods.

        images = [

            (
                "Original",
                image_data.closed_image
                or image_data.grayscale_image
                or image_data.image,
            ),

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
                "Final Binary",
                image_data.binary_image,
            ),
        ]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(16, 10),
        )

        axes = axes.flatten()

        for ax, (title, image) in zip(
            axes,
            images,
        ):

            ax.axis("off")

            if image is None:
                continue

            cmap = (
                "gray"
                if len(image.shape) == 2
                else None
            )

            ax.imshow(
                image,
                cmap=cmap,
            )

            ax.set_title(
                title,
                fontsize=11,
            )

        plt.tight_layout()

        plt.show()

    @staticmethod
    def before_after(
        before,
        after,
        before_title="Before",
        after_title="After",
    ) -> None:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
        )

        axes[0].imshow(
            before,
            cmap="gray",
        )

        axes[0].set_title(before_title)

        axes[0].axis("off")

        axes[1].imshow(
            after,
            cmap="gray",
        )

        axes[1].set_title(after_title)

        axes[1].axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def quality_table(
        image_data: ImageData,
    ) -> None:
        
        # Display threshold quality metrics.

        images = {

            "Global":
                image_data.global_threshold_image,

            "Otsu":
                image_data.otsu_image,

            "Adaptive Mean":
                image_data.adaptive_mean_image,

            "Adaptive Gaussian":
                image_data.adaptive_gaussian_image,

            "Binary":
                image_data.binary_image,
        }

        print()

        print("=" * 75)

        print(
            "{:<22} {:>12} {:>15} {:>15}".format(
                "Method",
                "Binary",
                "Foreground",
                "Components",
            )
        )

        print("=" * 75)

        for name, image in images.items():

            if image is None:
                continue

            valid = ThresholdValidator.validate(
                image
            )

            ratio = (
                ThresholdValidator
                .foreground_ratio(
                    image
                )
            )

            components = (
                ThresholdValidator
                .connected_components(
                    image
                )
            )

            print(
                "{:<22} {:>12} {:>15.4f} {:>15}".format(
                    name,
                    str(valid),
                    ratio,
                    components,
                )
            )

        print("=" * 75)

    @staticmethod
    def histogram(
        image_data: ImageData,
    ) -> None:
        
        # Histogram of binary image.

        image = image_data.binary_image

        if image is None:
            return

        plt.figure(
            figsize=(8, 5),
        )

        plt.hist(
            image.ravel(),
            bins=256,
        )

        plt.title(
            "Binary Histogram",
        )

        plt.xlabel(
            "Pixel Value",
        )

        plt.ylabel(
            "Frequency",
        )

        plt.tight_layout()

        plt.show()

    @staticmethod
    def best_result(
        image_data: ImageData,
    ) -> None:
        
        # Display final binary image.

        if image_data.binary_image is None:
            return

        plt.figure(
            figsize=(8, 8),
        )

        plt.imshow(
            image_data.binary_image,
            cmap="gray",
        )

        plt.title(
            "Best Threshold Result",
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> None:
        
        # Print statistics.

        ThresholdComparison.quality_table(
            image_data
        )

    @staticmethod
    def visualize_all(
        image_data: ImageData,
    ) -> None:
        
        # Display every visualization.

        ThresholdComparison.compare(
            image_data
        )

        ThresholdComparison.histogram(
            image_data
        )

        ThresholdComparison.quality_table(
            image_data
        )

        ThresholdComparison.best_result(
            image_data
        )