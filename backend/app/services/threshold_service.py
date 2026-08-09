from __future__ import annotations

import time

from app.models.image_data import ImageData

from app.preprocessing.global_threshold import GlobalThreshold
from app.preprocessing.otsu_threshold import OtsuThreshold
from app.preprocessing.adaptive_threshold import AdaptiveThreshold
from app.preprocessing.binary_refiner import BinaryRefiner
from app.preprocessing.threshold_validator import ThresholdValidator

from app.utils.image_statistics import ImageStatistics
from app.utils.exceptions import ImageProcessingError
from app.utils.logger import logger


class ThresholdService:
    
    # Threshold processing pipeline.

    @staticmethod
    def process(
        image_data: ImageData,
        use_global: bool = False,
        use_otsu: bool = False,
        use_adaptive_mean: bool = False,
        use_adaptive_gaussian: bool = True,
        refine: bool = True,
    ) -> ImageData:

        try:

            logger.info(
                "Starting threshold pipeline."
            )

            start = time.perf_counter()

            if use_global:

                GlobalThreshold.apply(image_data, inverse=True)

                image_data.binary_image = (
                    image_data.global_threshold_image
                )

            if use_otsu:

                OtsuThreshold.apply(image_data, inverse=True)

                image_data.binary_image = (
                    image_data.otsu_image
                )

            if use_adaptive_mean:

                AdaptiveThreshold.mean(image_data, inverse=True)

                image_data.binary_image = (
                    image_data.adaptive_mean_image
                )

            if use_adaptive_gaussian:

                AdaptiveThreshold.gaussian(image_data, inverse=True)

                image_data.binary_image = (
                    image_data.adaptive_gaussian_image
                )

            if refine:

                BinaryRefiner.refine(image_data)

            elapsed = (
                time.perf_counter() - start
            )

            logger.info(
                f"Threshold pipeline completed "
                f"in {elapsed:.3f} seconds."
            )

            return image_data

        except Exception as ex:

            logger.exception(ex)

            raise ImageProcessingError(
                str(ex)
            ) from ex

    @staticmethod
    def statistics(
        image_data: ImageData,
    ) -> dict:

        if image_data.binary_image is None:

            raise ImageProcessingError(
                "Binary image not available."
            )

        return ImageStatistics.basic(
            image_data.binary_image
        )

    @staticmethod
    def validation(
        image_data: ImageData,
    ) -> dict:

        return ThresholdValidator.evaluate(
            image_data
        )

    @staticmethod
    def latest_image(
        image_data: ImageData,
    ):

        return image_data.binary_image

    @staticmethod
    def processing_history(
        image_data: ImageData,
    ) -> list[str]:

        history = []

        for stage in (

            "Global Threshold",

            "Otsu Threshold",

            "Adaptive Mean",

            "Adaptive Gaussian",

            "Binary Refinement",
        ):

            if stage in image_data.processing_history:

                history.append(stage)

        return history

    @staticmethod
    def reset(
        image_data: ImageData,
    ):

        GlobalThreshold.reset(image_data)

        OtsuThreshold.reset(image_data)

        AdaptiveThreshold.reset(image_data)

        BinaryRefiner.reset(image_data)

        image_data.binary_image = None

        logger.info(
            "Threshold pipeline reset."
        )

    @staticmethod
    def is_processed(
        image_data: ImageData,
    ) -> bool:

        return image_data.binary_image is not None

    @staticmethod
    def preview(
        image_data: ImageData,
    ) -> dict:

        return {

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