import xml.etree.ElementTree
import time

class drug_data_reader():

    xml_file_name = 'full database.xml'

    def __init__(self,file_name,zipped=True):


        self.file_name=file_name
        self.zipped=zipped
        if zipped==False:
            assert False, 'didnt implement yet unzipped handling' #if i will need it, just read to file instead of unzipping it

        self.all_drugs = []
        self.drug_to_interactions = {}
        self.drug_to_type = {}
        self.drug_id_to_name = {}
        self.drug_id_to_groups = {}
        self.drug_id_to_targets = {}
        self.drug_id_to_enzymes = {}
        self.drug_id_to_carriers = {}
        self.drug_id_to_transporters = {}

        self.drug_id_to_ATC={}
        self.drug_id_to_cat={}
        self.drug_id_to_approved={}
        self.drug_id_to_weight={}
        self.atc_to_text = {}
        self.drug_id_to_smiles={}

        self.drug_id_to_tax_description={}
        self.drug_id_to_tax_direct_parent = {}
        self.drug_id_to_tax_kingdom = {}
        self.drug_id_to_tax_superclass = {}
        self.drug_id_to_tax_class = {}
        self.drug_id_to_tax_subclass = {}

    def read_data_from_file(self):
        print('reading file')
        start_time = time.time()

        if self.zipped:
            import zipfile
            archive = zipfile.ZipFile(self.file_name, 'r')
            db_file = archive.open(drug_data_reader.xml_file_name)


        root = xml.etree.ElementTree.parse(db_file).getroot()
        elapsed_time = time.time() - start_time
        ns = '{http://www.drugbank.ca}'
        for i, drug in enumerate(root):

            assert drug.tag == '{ns}drug'.format(ns=ns)
            drug_p_id = drug.findtext("{ns}drugbank-id[@primary='true']".format(ns=ns))
            assert drug_p_id not in self.all_drugs

            try:
                drug_type = drug.attrib['type']
            except:
                drug_type = None
            self.drug_to_type[drug_p_id] = drug_type
            assert len(drug.findall("{ns}groups".format(ns=ns))) == 1
            for i,group in enumerate(drug.findall("{ns}groups".format(ns=ns))[0].getchildren()):
                self.drug_id_to_groups.setdefault(drug_p_id,set()).add(group.text)
            self.drug_id_to_groups.setdefault(drug_p_id, set())
            for i,cat in enumerate(drug.findall("{ns}categories".format(ns=ns))[0].getchildren()):
                cat_text=cat.findtext('{ns}category'.format(ns=ns))
                self.drug_id_to_cat.setdefault(drug_p_id, set()).add(cat_text)
            self.drug_id_to_cat.setdefault(drug_p_id, set())



            mass = drug.findall("{ns}average-mass".format(ns=ns)) #resides in the main data about the drug
            weight=None
            if len(mass)>0:
                assert len(mass)==1,"more than 1 mass found"
                weight = float(mass[0].text)
                self.drug_id_to_weight[drug_p_id]=weight


            self.all_drugs.append(drug_p_id)
            self.drug_id_to_name[drug_p_id] = drug.findtext("{ns}name".format(ns=ns))
            assert len(drug.findall("{ns}drug-interactions".format(ns=ns))) == 1
            for interaction in drug.findall("{ns}drug-interactions".format(ns=ns))[0].getchildren():
                # interaction.findtext('{ns}name'.format(ns=ns))
                other_drug_id = interaction.findtext('{ns}drugbank-id'.format(ns=ns))
                interaction_description = interaction.findtext('{ns}description'.format(ns=ns))
                #self.drug_to_interactions.setdefault(drug_p_id, set()).add((other_drug_id,interaction_description))
                self.drug_to_interactions.setdefault(drug_p_id, set()).add((other_drug_id))

            enzymes = []
            targets = []
            carriers = []
            transporters = []

            for t in drug.iter("{ns}target".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        targets.append(gn.text)
            for t in drug.iter("{ns}enzymes".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        enzymes.append(gn.text)
            for t in drug.iter("{ns}carriers".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        carriers.append(gn.text)
            for t in drug.iter("{ns}transporters".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        transporters.append(gn.text)
            self.drug_id_to_targets[drug_p_id]=set(targets)
            self.drug_id_to_enzymes[drug_p_id]=set(enzymes)
            self.drug_id_to_carriers[drug_p_id] = set(carriers)
            self.drug_id_to_transporters[drug_p_id] = set(transporters)



            tax = drug.find("{ns}classification".format(ns=ns))
            try:
                self.drug_id_to_tax_description[drug_p_id] =  tax.find("{ns}description".format(ns=ns)).text
            except:
                self.drug_id_to_tax_description[drug_p_id]=None
            try:
                self.drug_id_to_tax_direct_parent[drug_p_id] = tax.find("{ns}direct-parent".format(ns=ns)).text
            except:
                self.drug_id_to_tax_direct_parent[drug_p_id] =None
            try:
                self.drug_id_to_tax_kingdom[drug_p_id] = tax.find("{ns}kingdom".format(ns=ns)).text
            except:
                self.drug_id_to_tax_kingdom[drug_p_id] =None
            try:
                self.drug_id_to_tax_superclass[drug_p_id]= tax.find("{ns}superclass".format(ns=ns)).text
            except:
                self.drug_id_to_tax_superclass[drug_p_id] =None
            try:
                self.drug_id_to_tax_class[drug_p_id] = tax.find("{ns}class".format(ns=ns)).text
            except:
                self.drug_id_to_tax_class[drug_p_id] = None
            try:
                self.drug_id_to_tax_subclass[drug_p_id] = tax.find("{ns}subclass".format(ns=ns)).text
            except:
                self.drug_id_to_tax_subclass[drug_p_id] = None


            if len(drug.findall("{ns}calculated-properties".format(ns=ns))) > 0: #TODO: add all of the properties
                for i, prop in enumerate(drug.findall("{ns}calculated-properties".format(ns=ns))[0].getchildren()):
                    if prop.find('{http://www.drugbank.ca}kind').text == 'SMILES':
                        smiles = prop.find('{ns}value'.format(ns=ns)).text
                        self.drug_id_to_smiles[drug_p_id] = smiles

            assert len(drug.findall("{ns}experimental-properties".format(ns=ns))) == 1
            for i,prop in enumerate(drug.findall("{ns}experimental-properties".format(ns=ns))[0].getchildren()):
                if prop.find('{http://www.drugbank.ca}kind').text=='Molecular Weight':
                    assert weight==None or float(prop.find('{ns}value'.format(ns=ns)).text)==weight,'found weight twice'
                    weight = float(prop.find('{ns}value'.format(ns=ns)).text)
                    self.drug_id_to_weight[drug_p_id] = weight





            assert drug_p_id not in self.drug_id_to_ATC
            self.drug_id_to_ATC[drug_p_id]=set()
            for atc in drug.findall("{ns}atc-codes".format(ns=ns))[0].getchildren():
                self.drug_id_to_ATC[drug_p_id].add(atc.get('code'))
                for atc_child in atc.getchildren():
                    code = atc_child.get('code')
                    text = atc_child.text
                    self.atc_to_text[code]=text
        print('time to read file:', elapsed_time)
        print('number of drugs read:', len(self.all_drugs))
        print('number of drugs with interactions', len(self.drug_to_interactions))
        print('drugs with no interactions:', len(set(self.all_drugs) - set(self.drug_to_interactions.keys())))
        print('groups (approved etc.)',{y for x in self.drug_id_to_groups.values() for y in x})
        print('number of drugs with weight:',len(self.drug_id_to_weight.keys()))
        print('number of drugs with targets:', len(self.drug_id_to_enzymes.keys()))
        print('number of drugs with enzymes:', len(self.drug_id_to_targets.keys()))
        print('Drug types:', set(self.drug_to_type.values()))

        #print([self.drug_id_to_name[x] for x in set(self.all_drugs) - set(self.drug_to_interactions.keys())])
