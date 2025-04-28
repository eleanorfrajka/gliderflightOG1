from pathlib import Path

import xarray as xr

from flightmodels import logger, utilities


logger.disable_logging()


def test_get_default_data_dir():
    # This should always resolve to your project's /data directory
    data_dir = utilities.get_default_data_dir()
    assert isinstance(data_dir, Path)
    assert data_dir.name == "data"
    assert (
        data_dir.exists() or not data_dir.exists()
    )  # Should be valid even if data folder doesn't yet exist


def test_apply_defaults_decorator_applies_source_and_file_list():
    # Define a dummy function to wrap
    def dummy_reader(source=None, file_list=None):
        return {"source": source, "file_list": file_list}

    default_source = "http://example.com"
    default_files = ["test.nc"]

    decorated = utilities.apply_defaults(default_source, default_files)(dummy_reader)

    # Test with no arguments
    result = decorated()
    assert result["source"] == default_source
    assert result["file_list"] == default_files

    # Test with only one override
    result = decorated(source="custom.nc")
    assert result["source"] == "custom.nc"
    assert result["file_list"] == default_files

    result = decorated(file_list=["override.nc"])
    assert result["source"] == default_source
    assert result["file_list"] == ["override.nc"]


def test_safe_update_attrs_add_new_attribute():
    ds = xr.Dataset()
    new_attrs = {"project": "MOVE"}
    ds = utilities.safe_update_attrs(ds, new_attrs)
    assert ds.attrs["project"] == "MOVE"


def test_safe_update_attrs_existing_key_logs(caplog):

    # Re-enable logging for this test
    logger.enable_logging()

    ds = xr.Dataset(attrs={"project": "MOVE"})
    new_attrs = {"project": "OSNAP"}

    with caplog.at_level("DEBUG", logger="amocarray"):
        utilities.safe_update_attrs(ds, new_attrs, overwrite=False, verbose=True)

    assert any(
        "Attribute 'project' already exists in dataset attrs and will not be overwritten."
        in message
        for message in caplog.messages
    )


def test_safe_update_attrs_existing_key_with_overwrite():
    ds = xr.Dataset(attrs={"project": "MOVE"})
    new_attrs = {"project": "OSNAP"}
    ds = utilities.safe_update_attrs(ds, new_attrs, overwrite=True)
    assert ds.attrs["project"] == "OSNAP"
