import xml.etree.ElementTree as ET
from lxml import etree
from itertools import islice, chain
import six
# from pymongo import MongoClient
import psycopg2
import re
import pandas as pd
import json
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import pyspark as ps
impor string


def cleanhtml_test(raw_html):
    code = re.compile('<pre><code>.*?</code></pre>')
    sans_code = re.sub(code, ' ', raw_html)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sans_code)
    return cleantext

def get_text_data_test(fp):
    df = pd.read_csv(fp, header=None)
    df.columns = ['postid', 'title', 'body' ]
    text = df["title"].apply(cleanhtml_test) + " "+ df["body"].apply(cleanhtml_test)
    return text

def normalize_text(text, stops):
    text = 

# fp = './clean_dev_set.csv/"part-00000-4df5cbe6-3d8e-44ad-a26c-42f2303b93a0-c000.csv"'
fp="/Users/dentonzhao/Projects/StackExchange-Analysis/clean_dev_set.csv/part-00000-4df5cbe6-3d8e-44ad-a26c-42f2303b93a0-c000.csv"
text = get_text_data_test(fp)




    

class Data_reader():
    """reads in xml file, converts said file into a raw pandas df"""

    def __init__(self,filepath):
        """parses the filepath if specified"""
        if filepath:
            self.read_xml(filepath)

    def read_xml(self, filepath):
        """
        parses xml filepath and stores the data as an ElementTree

        INPUTS:
        -------
        file: string
        """
        self.filepath = filepath
        parsed_data = ET.parse(self.filepath)
        self.data = parsed_data.getroot()

    def tag_cleaner(string):
        """
        removes the '<' and '>' characters from string and converts into listself.
        if the string is empty, returns an empty list
        i.e. <tag1><tag2> => [tag1, tag2]
        """
        if string:
            cleaned_tag = re.sub(r'<','', string)
            tags = cleaned_tag.split('>')[:-1]
        else:
            tags = [None]
        return(tags)


    # def get_schema(self, schema_file):
    #     """
    #     will get schema files (this is currently from csv)
    #     from there abstract the index and the datatype
    #
    #     """
    #     schema = pd.read_csv(schema_file)
    #     # remove the ' (PK)' suffix from the column_name
    #     schema['column_name'].iloc[0] = schema['column_name'].iloc[0][0:-5]
    #     schema[['column_name', 'data_type']]
    #     self.schema = pd.Series(data = schema['data_type'].values, index=schema['column_name'])

    # def infer_schema(self):
    #     """
    #     given a dataframe and schema, force datatypes to match schema
    #     """
    #
    #     self.schema
    #     sql_to_python = json.load(open('schema.json'))
    #     columns = self.master_data.columns
    #     column_to_dtype ={}
    #     for column in columns:
    #         column_to_dtype[column] = sql_to_python[self.schema[column]]
    #     dataframe = self.master_data.astype(column_to_dtype)
    #     return(dataframe)

    def make_dataframe(self, force_schema = True):
       """
       from xml data, schema series, makes a pandas df. posts with more than one tag are
       duplicated to form multiple rows
       """

       data_columns = ['PostTypeId','Body','Tags']

       all_data =[]

       for row in self.data:
           #row_data is an empty dict, populate row with cleaned_values
           row_data ={}
           for column in data_columns:
               row_data[column] = row.attrib.get(column)
           all_data.append(row_data)
       self.master_data = pd.DataFrame(data=all_data,columns=data_columns)
       self.clean_df()
       return(self.master_data)

    def clean_df(self):
        self.master_data['PostTypeId'] = self.master_data['PostTypeId'].astype(int)

class Data_cleaner():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', raw_html)
        return cleantext

    def clean_dataframe(self):
        self.dataframe['Body'] = self.dataframe['Body'].apply(self.cleanhtml)
        self.dataframe['Tags'] = self.dataframe['Tags'].apply(self.clean_tags)

    def clean_tags(self, string):
        if string:
            cleaned_tag = re.sub(r'<','', string)
            tags = cleaned_tag.split('>')[:-1]
        else:
            tags = [None]
        return(tags)

class word_tokenizer():
    def __init__(self):
        pass
