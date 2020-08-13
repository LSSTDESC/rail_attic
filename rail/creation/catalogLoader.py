import GCRCatalogs
import yaml
import pandas as pd


class catalogLoader():

    def __init__(self, config_file):

        self.config_data = self.read_config_yaml(config_file)
        self.gcr_catalog = GCRCatalogs.load_catalog(self.config_data['catalog_name'])

        return

    def read_config_yaml(self, filename):

        with open(filename, 'r') as stream:
            config_data = yaml.safe_load(stream)

        return config_data

    def get_catalog_mags_and_redshift(self):

        cat_columns = [x for x in self.config_data['mag_columns']]
        cat_columns.append(self.config_data['redshift_column'])
        gcr_filters = self.config_data['gcr_filters']
        native_filters = self.config_data['native_filters']
        data = self.gcr_catalog.get_quantities(cat_columns, filters=gcr_filters,
                                               native_filters=native_filters)

        data_df = pd.DataFrame(data)

        return data_df
