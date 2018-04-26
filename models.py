"""
Stuff to build models with

"""
import pandas as pd
from bs4 import BeautifulSoup



# print(text)

"""
This section of code will be for setting up the dev test environment.
"""
def create_dev_set(infile, outfile):
    """
    from filepath create a 1000 row sample file to output_file
    INPUTS:
    infile = string location of csv to read
    outfile = string location of the csv to writer

    OUTPUT:
    outfile
    """
    data = pd.read_csv(infile, nrows=2000, encoding='utf8')
    data.to_csv(outfile, encoding = 'utf8')

def html_cleaner(text):
    """
    Takes in a string, strips away the code and returns the cleaned text body and the code body

    Inputs:
    text: string to clean

    Output:
    tuple containing the clean text and the cleaned code
    """
    soup = BeautifulSoup(text, 'html.parser')
    code_section = [code.text for code in soup.find_all('code')]
    for code in soup.find_all('code'):
        soup.code.decompose()
    text_section = soup.get_text()
    return(text_section, code_section)

# data = pd.read_csv('csvs/dev_set.csv', nrows=1000, encoding = 'utf8', doublequote=False, escapechar='\\')
