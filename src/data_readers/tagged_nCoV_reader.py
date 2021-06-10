import os

import pandas as pd

class nCoV_tagged_data_reader():
    drug_id_to_name = {
        'DB00207': 'Azithromycin',
        'DB01611': 'Hydroxychloroquine (Plaquenil)  200mg tab',
        'DB00608': 'Chloroquine 250 mg tab',
        'DB14761': 'Remedesivir',
        'DB01601': 'Lopinavir (Kaletra)',
        'DB00503': 'Ritonavir (Kaletra)',
        'DB13997': 'Baloxavir marboxil',
        'DB12466': 'Favipiravir',
        'DB06273': 'Tocilizumab',
        'DB00060': 'Interferon beta-1a',
        # 'null':'Hyperimmuneglobulin - does not exist',
        'DB01264': 'Darunavir Rezolsta',
        'DB09065': 'Cobicistat Rezolsta',
        'DB00105': 'Interferon α2B (Interferon alfa-2b)',
        'DB11767': 'Sarilumab',  # https://rotter.net/forum/scoops1/618965.shtml
        'DB01593': 'Zinc',
        'DB00602': 'Ivermectin',  # https://www.sciencedirect.com/science/article/pii/S0166354220302011
        'DB12610': 'Ebselen',  # paper from lior nature
        'DB00822': 'Disulfiram',  # paper from lior nature
        'DB12129': 'Tideglusib',  # paper from lior nature
        'DB09010': 'Carmofur',  # paper from lior nature
        'DB05448': 'PX-12',  # paper from lior nature
        'DB11703':'Calquence', #https://www.standard.co.uk/business/glaxo-and-astrazeneca-in-covid19-breakthrough-hopes-a4413876.html
        'DB13609':'Umifenovir', #https://www.visualcapitalist.com/every-vaccine-treatment-covid-19-so-far/
        'DB11779' : "Danoprevir",#same
        #'DB06273': 'Tocilizumab' #same
        'DB15148':'Lenzilumab',  #same
        #'DB01264':'Darunavir' #same
        #'DB09065':'Cobicistat'#same
        'DB01394':'Colchicine', #same
        #'DB11767':'Kevzara'#same
        'DB00112':'Bevacizumab', #same
        #'DB14761':''#same
        'DB05941':'Leronlimab',#same
        'DB06260':'Aviptadil',#same
        #IFN-β cant find
        'DB08868':'Fingolimod',#same
        'DB14776':'Camrelizumab',#same
        #mesenchymal cells cant find
        'DB00678':'Losartan',#same
        'DB15280':'Gimsilumab',#same
        'DB09036':'Siltuximab',#same
        #Plasmapheresis cant find - not a drug.
        'DB11817':'Baricitinib', #https://www.drugbank.ca/drugs/DB11817
        'DB00927': 'Famotidine', #https://www.drugtopics.com/latest/famotidine-trial-underway-nyc-covid-19-treatment


    }

    def read_zhou_et_al_tagged(self):
        tagged_drugs_path=os.path.join('pickles', 'data', 'nCoV','Drug_screening.xlsx') #r'C:\Users\Administrator\PycharmProjects\DDI_SMC\data\WHO essential med classification.xlsx'
        tagged_drugs = pd.read_excel(tagged_drugs_path)
        tagged_drugs = tagged_drugs.dropna(subset=['drugBank_id'])
        tagged_drugs['drugBank_id'] =tagged_drugs['drugBank_id'].astype(str)
        tagged_drugs=tagged_drugs[['drugBank_id','In_Final_List']]
        for drug_id in ['DB00608','DB00959','DB12466','DB01611','DB00207','DB14761','DB00028']:
            tagged_drugs.loc[tagged_drugs['drugBank_id']==drug_id,'In_Final_List']='Yes'
        #might help: DB00608 (Chloroquine) # https://www.sciencemag.org/news/2020/03/who-launches-global-megatrial-four-most-promising-coronavirus-treatments
        #DB00959 (methylprednisolone)
        #DB12466 (Favipiravir)
        #DB01611 (Hydroxychloroquine) or Plaquenil #news 12 # https://www.sciencedirect.com/science/article/pii/S0924857920300996?via%3Dihub# https://www.sciencemag.org/news/2020/03/who-launches-global-megatrial-four-most-promising-coronavirus-treatments
        #DB00207 (Azithromycin) or Azenil #news 12
        #DB14761 (Remdesivir) not approved # from lior
        #DB00028  (Human immunoglobulin G) #from FDA
        # DB00503   Ritonavir Or Kaletra #from ynet patent
        #lopinavir and ritonavir # https://www.sciencemag.org/news/2020/03/who-launches-global-megatrial-four-most-promising-coronavirus-treatments
        #interferon-beta
        return tagged_drugs


