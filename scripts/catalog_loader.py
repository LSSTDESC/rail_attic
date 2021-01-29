# Script to load photometry and redshift from an
# extragalactic catalog accessible through GCRCatalogs.

import argparse
import GCRCatalogs
import yaml
import pandas as pd


def read_config_yaml(filename):
    """
    Read in the configuration file.
    For an example file see examples/configs/buzzardCatLoader.yaml

    Parameters
    ----------
    filename : str
        Filepath of configuration file.

    Returns
    -------
    dict
        Dictionary with parameters to pass to GCRCatalogs.
    """

    with open(filename, 'r') as stream:
        config_data_dict = yaml.safe_load(stream)

    return config_data_dict


def get_catalog_mags_and_redshift(gcr_catalog, config_data_dict):
    """
    Use the gcrCatalog object for a given catalog and the
    dictionary with the configuration data to return the
    catalog data as a pandas dataframe.

    Parameters
    ----------
    gcr_catalog : GCRCatalog object
        GCRCatalog object set up for the desired catalog.

    config_data_dict : dict
        Configuration data dictionary from `read_config_yaml`.

    Returns
    -------
    pandas dataframe
        Magnitude and redshift information from the catalog.
    """

    cat_columns = [x for x in config_data_dict['mag_columns']]
    cat_columns.append(config_data_dict['redshift_column'])
    gcr_filters = config_data_dict['gcr_filters']
    native_filters = config_data_dict['native_filters']
    data = gcr_catalog.get_quantities(cat_columns, filters=gcr_filters,
                                      native_filters=native_filters)

    data_df = pd.DataFrame(data)

    return data_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Path to catalog config file.")
    parser.add_argument("catalog_out", type=str,
                        help="Path to save catalog as pandas dataframe.")
    args = parser.parse_args()

    config_data_dict = read_config_yaml(args.config_file)
    gcr_catalog = GCRCatalogs.load_catalog(config_data_dict['catalog_name'])

    catalog_df = get_catalog_mags_and_redshift(gcr_catalog, config_data_dict)
    catalog_df.to_csv(args.catalog_out, index=False)
