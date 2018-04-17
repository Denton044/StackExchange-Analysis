import xml.etree.ElementTree as ET
# from pymongo import MongoClient
import psycopg2
import re
import pandas as pd
import json

# sample_file = ('startups.stackexchange.com/Tags.xml')


def read_xml(file):
    """
    parses xml filepath and stores the data as an ElementTree

    INPUTS:
    -------
    file: string

    OUTPUTS:
    --------
    data: Element Tree type file
    """
    parsed_data = ET.parse(file)
    root = parsed_data.getroot()
    return root

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

def get_schema(schema_file):
    """
    will get schema files (this is currently from csv)
    from there abstract the index and the datatype

    """
    schema = pd.read_csv(schema_file)
    # remove the ' (PK)' suffix from the column_name
    schema['column_name'].iloc[0] = schema['column_name'].iloc[0][0:-5]
    schema[['column_name', 'data_type']]
    schema = pd.Series(data = schema['data_type'].values,
                       index=schema['column_name'])
    return(schema)


def make_post_dataframe(data, schema, col_to_clean = None):
    """
    from xml data, schema series, makes a pandas df. posts with more than one tag are
    duplicated to form multiple rows
    """

    if col_to_clean:
        data_columns = schema.index.drop(col_to_clean)
    else:
        data_columns = schema.index

    all_data =[]

    for row in data:
        raw_tags = row.attrib.get(col_to_clean)
        tags = tag_cleaner(raw_tags)
        #row_data is an empty dict, populate row with cleaned_values
        row_data ={}
        for column in data_columns:
            row_data[column] = row.attrib.get(column)

        for tag in tags:
            cleaned_row = {col_to_clean:tag}
            row_data = {**row_data, **cleaned_row}
            all_data.append(row_data)

    master_data = pd.DataFrame(
                               data=all_data,
                               columns=schema.index)
    return(master_data)


def infer_schema(dataframe, schema):
    """
    given a dataframe and schema, force datatypes to match schema
    """
    sql_to_python = json.load(open('schema.json'))
    columns = dataframe.columns
    column_to_dtype ={}
    for column in columns:
        column_to_dtype[column] = sql_to_python[schema[column]]
    dataframe = dataframe.astype(column_to_dtype)
    return(dataframe)




def make_postgres_table(data, keys, table_name, database_name):
    """
    inputs an element tree data structure and then makes makes a postgres table for a given database
    """

    pconn = psycopg2.connect(
                            dbname = '{}'.format(database_name),
                            user = 'postgres', host ='localhost')
    pconn.cursor.execute('')
    pcur= pconn.cursor()

    sql_statement = """ INSERT INTO \
                    {} (Id, TagName, Count, ExcerptPostId, WikiPostId) VALUES \
                    ('{}','{}','{}','{}','{}')""".format(table_name, *keys)

    for child in data[0:5]:
        print('Inserting \n {} \n'.format(child))
        pcur.execute(sql_statement)
    pconn.commit()

if __name__ == "__main__":
    sample_file = ('startups.stackexchange.com/Posts.xml')
    schema = get_schema("QueryResults.csv")
    data = read_xml(sample_file)
    df = make_post_dataframe(
                            data,
                            schema,
                            col_to_clean = 'Tags'
                            )
    df = infer_schema(df, schema)
    print(df.dtypes)

    #
    # make_postgres_table(data, column_names, 'Tags', 'startups.stackexchange')
