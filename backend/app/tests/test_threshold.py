from pathlib import Path

import pytest

from app.services.image_service import ImageService
from app.services.threshold_service import ThresholdService

from app.preprocessing.global_threshold import GlobalThreshold
from app.preprocessing.otsu_threshold import OtsuThreshold
from app.preprocessing.adaptive_threshold import AdaptiveThreshold
from app.preprocessing.binary_refiner import BinaryRefiner
from app.preprocessing.threshold_validator import ThresholdValidator

from app.utils.exceptions import ImageProcessingError


TEST_IMAGE = Path(
    "data/images/1.jpeg"
)


@pytest.fixture
def image_data():

    return ImageService.load_image(TEST_IMAGE)


# =========================================================
# Global Threshold
# =========================================================

def test_global_threshold(image_data):

    result = GlobalThreshold.apply(image_data)

    assert result is not None

    assert image_data.global_threshold_image is not None


def test_global_statistics(image_data):

    GlobalThreshold.apply(image_data)

    stats = GlobalThreshold.statistics(image_data)

    assert stats["Width"] > 0

    assert stats["Height"] > 0


# =========================================================
# Otsu Threshold
# =========================================================

def test_otsu_threshold(image_data):

    result = OtsuThreshold.apply(image_data)

    assert result is not None

    assert image_data.otsu_image is not None


def test_otsu_statistics(image_data):

    OtsuThreshold.apply(image_data)

    stats = OtsuThreshold.statistics(image_data)

    assert stats["Width"] > 0

    assert "Threshold" in stats


# =========================================================
# Adaptive Mean
# =========================================================

def test_adaptive_mean(image_data):

    result = AdaptiveThreshold.mean(image_data)

    assert result is not None

    assert image_data.adaptive_mean_image is not None


# =========================================================
# Adaptive Gaussian
# =========================================================

def test_adaptive_gaussian(image_data):

    result = AdaptiveThreshold.gaussian(image_data)

    assert result is not None

    assert image_data.adaptive_gaussian_image is not None


# =========================================================
# Binary Refinement
# =========================================================

def test_binary_refiner(image_data):

    AdaptiveThreshold.gaussian(image_data)

    result = BinaryRefiner.refine(image_data)

    assert result is not None

    assert image_data.binary_image is not None


def test_remove_noise(image_data):

    AdaptiveThreshold.gaussian(image_data)

    result = BinaryRefiner.remove_noise(image_data)

    assert result is not None


def test_fill_holes(image_data):

    AdaptiveThreshold.gaussian(image_data)

    result = BinaryRefiner.fill_holes(image_data)

    assert result is not None


# =========================================================
# Threshold Validator
# =========================================================

def test_validator(image_data):

    AdaptiveThreshold.gaussian(image_data)

    BinaryRefiner.refine(image_data)

    assert ThresholdValidator.validate(
        image_data.binary_image
    )


def test_foreground_ratio(image_data):

    AdaptiveThreshold.gaussian(image_data)

    BinaryRefiner.refine(image_data)

    ratio = ThresholdValidator.foreground_ratio(
        image_data.binary_image
    )

    assert ratio >= 0

    assert ratio <= 1


def test_background_ratio(image_data):

    AdaptiveThreshold.gaussian(image_data)

    BinaryRefiner.refine(image_data)

    ratio = ThresholdValidator.background_ratio(
        image_data.binary_image
    )

    assert ratio >= 0

    assert ratio <= 1


def test_connected_components(image_data):

    AdaptiveThreshold.gaussian(image_data)

    BinaryRefiner.refine(image_data)

    count = ThresholdValidator.connected_components(
        image_data.binary_image
    )

    assert count >= 0


# =========================================================
# Threshold Service
# =========================================================

def test_threshold_service(image_data):

    result = ThresholdService.process(image_data)

    assert result.binary_image is not None


def test_service_statistics(image_data):

    ThresholdService.process(image_data)

    stats = ThresholdService.statistics(image_data)

    assert stats["Width"] > 0


def test_service_validation(image_data):

    ThresholdService.process(image_data)

    report = ThresholdService.validation(image_data)

    assert report["Binary"] is True


def test_processing_history(image_data):

    ThresholdService.process(image_data)

    history = ThresholdService.processing_history(image_data)

    assert len(history) > 0


def test_preview(image_data):

    ThresholdService.process(image_data)

    preview = ThresholdService.preview(image_data)

    assert isinstance(preview, dict)

    assert "Binary" in preview


def test_latest_image(image_data):

    ThresholdService.process(image_data)

    assert ThresholdService.latest_image(image_data) is not None


def test_processed(image_data):

    assert not ThresholdService.is_processed(image_data)

    ThresholdService.process(image_data)

    assert ThresholdService.is_processed(image_data)


def test_reset(image_data):

    ThresholdService.process(image_data)

    ThresholdService.reset(image_data)

    assert image_data.binary_image is None

    assert image_data.global_threshold_image is None

    assert image_data.otsu_image is None

    assert image_data.adaptive_mean_image is None

    assert image_data.adaptive_gaussian_image is None


# =========================================================
# Error Handling
# =========================================================

def test_none_image():

    with pytest.raises(ImageProcessingError):

        ThresholdValidator.validate(None)


def test_invalid_image():

    with pytest.raises(Exception):

        ImageService.load_image(
            "invalid/image.jpg"
        )


# =========================================================
# Multiple Runs
# =========================================================

def test_multiple_processing(image_data):

    for _ in range(5):

        ThresholdService.process(image_data)

    assert image_data.binary_image is not None


# =========================================================
# Binary Image
# =========================================================

def test_binary_values(image_data):

    ThresholdService.process(image_data)

    values = set(
        image_data.binary_image.flatten()
    )

    assert values.issubset({0, 255})


def test_binary_dimensions(image_data):

    ThresholdService.process(image_data)

    assert (
        image_data.binary_image.shape ==
        image_data.image.shape[:2]
    )