import pytest

from flightmodels import logger, readers

logger.disable_logging()


def test_load_sample_dataset_invalid_array():
    with pytest.raises(
        ValueError,
        match="Sample dataset for array 'invalid' is not defined",
    ):
        readers.load_sample_dataset("invalid")
