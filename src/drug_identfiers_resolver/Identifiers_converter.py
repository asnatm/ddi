import pubchempy as pcp
import pandas as pd
from os import path
import json
import urllib
import csv


DRUG_BANK_NAME_SEARCH_URL = f'https://www.drugbank.ca/unearth/q?utf8=%E2%9C%93&query=drug_name&searcher=drugs'
PUBCHEM_SEARCH_BY_NAME_URL = \
    f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/drug_name/xrefs/RegistryID,RN,PubMedID/JSONP'
PUBCHEM_SEARCH_BY_CID_URL = \
    f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/cid_value/xrefs/RegistryID,RN,PubMedID/JSONP'


class Identifiers_converter():

    def __init__(self, csvs_dir=''):
        self.csvs_dir = csvs_dir

    def get_sids_by_name(self, drug_name):
        drug_name = self.process_query(drug_name)
        csv_path = self.csvs_dir + 'sides.csv'
        exists, result = self.check_query_in_file(csv_path, drug_name)
        if exists:
            return result
        else:
            result = pcp.get_sids(drug_name, 'name')
            sids = result[0]['SID']
            self.add_query_to_file(csv_path, drug_name, sids)
            return sids

    def get_aids_by_name(self, drug_name):
        drug_name = self.process_query(drug_name)
        csv_path = self.csvs_dir + 'aids.csv'
        exists, result = self.check_query_in_file(csv_path, drug_name)
        if exists:
            return result
        else:
            result = pcp.get_aids(drug_name, 'name')
            aids = result[0]['AID']
            self.add_query_to_file(csv_path, drug_name, aids)
            return aids

    def get_cids_by_name(self, drug_name):
        drug_name = self.process_query(drug_name)
        csv_path = self.csvs_dir + 'cides.csv'
        exists, result = self.check_query_in_file(csv_path, drug_name)
        if exists:
            return result
        else:
            result = pcp.get_cids(drug_name, 'name')
            cids = result[0]['CID']
            self.add_query_to_file(csv_path, drug_name, cids)
            return cids

    def get_all_ids(self, drug_name):
        sids = self.get_sids_by_name(drug_name)
        aids = self.get_aids_by_name(drug_name)
        # cids = get_cids_by_name(drug_name)

    def get_cid_and_dbid_by_name(self, drug_name):
        """
        Return both CID and DrugBank ID of a drug
        :param drug_name: drug name
        :return: Dictionary that contains CID (key is 'CID') and DrugBank ID (key is 'DB_ID')
        """
        drug_name = self.process_query(drug_name)
        url = PUBCHEM_SEARCH_BY_NAME_URL.replace('drug_name', str(drug_name))
        response = urllib.request.urlopen(url)
        json_string = response.read().decode()[9:-3]
        data = json.loads(json_string)
        registry_ids = data['InformationList']['Information'][0]['RegistryID']
        cid = data['InformationList']['Information'][0]['CID']
        for id in registry_ids:
            if id[:2] == 'DB':
                return {'CID': cid, 'DB_ID': id}

    @staticmethod
    def get_drug_bank_id_by_cid(cid):
        """
        Returns the DrugBank ID of a given CID
        :param cid: CID
        :return: DrugBank ID
        """
        url = PUBCHEM_SEARCH_BY_CID_URL.replace('cid_value', str(cid))
        response = urllib.request.urlopen(url)
        json_string = response.read().decode()[9:-3]
        data = json.loads(json_string)
        registry_ids = data['InformationList']['Information'][0]['RegistryID']
        for id in registry_ids:
            if id[:2] == 'DB':
                return id

    def get_drug_bank_code_by_name(self, drug_name):
        """
        Returns drugbank drug code of a given drug.
        The function check first if the drug code exists in the saved file, if not checks with drugbank
        :param drug_name: name of the drug to search
        :return: drugbank drug code
        """
        drug_name = self.process_query(drug_name)
        csv_path = self.csvs_dir + 'drugbank_codes.csv'
        exists, result = self.check_query_in_file(csv_path, drug_name)
        if exists:
            return result
        else:
            drug_code = self._retrieve_from_drugbank(drug_name)
            if drug_code[:2] == 'DB':
                self.add_query_to_file(csv_path, drug_name, drug_code)
                return drug_code
            else:
                return 'drug not found'

    def _retrieve_from_drugbank(self, drug_name):
        """
        Search drugbank for the drug code
        :param drug_name: drug to be searched
        :return: drug code
        """
        drug_name = self.process_query(drug_name)
        url = DRUG_BANK_NAME_SEARCH_URL.replace('drug_name', str(drug_name))
        response = urllib.request.urlopen(url)
        drug_code = response.url[response.url.rfind('/') + 1:]
        return drug_code

    @staticmethod
    def process_query(query):
        modified_query = query.strip()
        modified_query = modified_query.strip('\t')
        return modified_query

    def check_query_in_file(self, file_path, query):
        csv_path = self.csvs_dir + f'{file_path}'
        if path.isfile(csv_path):
            drugs_frame = pd.read_csv(csv_path)
        else:
            drugs_frame = pd.DataFrame.from_dict({'query': [], 'result': []})
            drugs_frame.to_csv(csv_path, index=False)

        if drugs_frame['query'].values:
            return True, drugs_frame[drugs_frame['query'] == query]['result'].iloc[0]
        else:
            return False, None

    @staticmethod
    def add_query_to_file(file_path, query, result):
        with open(file_path, 'a') as file:
            CSVWriter = csv.writer(file)
            row = [query, result]
            CSVWriter.writerow(row)

# Example:
# id_converter = Identifiers_converter()
# print(id_converter.get_cid_and_dbid_by_name('aspirin'))
# print(id_converter.get_drug_bank_code_by_name('aspirin'))
# print(id_converter.get_sids_by_name('aspirin'))
# print(id_converter.get_drug_bank_id_by_cid(2244))