from bs4 import BeautifulSoup

# -- load in text -- 

def txt_load_str(path, encoding='utf-8', join_char=' '):
    """Load a txt file to string"""
    
    text = txt_load(path, encoding=encoding)
            
    return join_char.join(text)

def txt_load(path, encoding='utf-8'):
    """Load a txt file to list of strings"""
    text = list()
    
    with open(path, 'r', encoding=encoding) as infile:
        for line in infile:
            text.append(line)
            
    return text

# -- text processing --

def is_html(string):
    """Check whether a string is html code"""
    return bool(BeautifulSoup(string, "html.parser").find())

def fix_unicode(string):
    """Fix the unicode on a given string"""
    return ftfy.fix_text(string)

def wordcount(string):
    """Count the number of words in a string"""
    count = len(re.findall(r'\w+', string))
    return count