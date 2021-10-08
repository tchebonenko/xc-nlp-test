# python

from typing import List
from lxml import etree 
import sklearn.feature_extraction.text
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer


# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_BIG_BILL = 'samples/congress/116/uslm/BILLS-116s1790enr.xml'

 
def getEnum(section) -> str:
  enumpath = section.xpath('enum')  
  if len(enumpath) > 0:
    return enumpath[0].text
  return ''

def getHeader(section) -> str:
  headerpath = section.xpath('header')  
  if len(headerpath) > 0:
    return headerpath[0].text
  return ''

def text_to_ngrams(txt: str , ngram_size: int = 4) -> list:
    """
    Gets ngrams from text
    """
    # See https://stackoverflow.com/a/32128803/628748
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(txt)
    #vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size),
    #    tokenizer=TreebankWordTokenizer().tokenize, lowercase=True)
    vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size),
        tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    vect.fit(sentences)
    ngrams = vect.get_feature_names_out()
    print('{1}-grams: {0}'.format(ngrams, ngram_size))
    # TODO get unt of tokens
    return ngrams


def xml_to_sections(xml_path: str):
    """
    Parses the xml file into sections 
    """
    try:
        billTree = etree.parse(xml_path)
    except:
        raise Exception('Could not parse bill')
    sections = billTree.xpath('//section')
    if len(sections) == 0:
        return []
    return [{
            'section_number': getEnum(section) ,
            'section_header':  getHeader(section),
            'section_text': etree.tostring(section, method="text", encoding="unicode"),
            'section_xml': etree.tostring(section, method="xml", encoding="unicode")
        } if (section.xpath('header') and len(section.xpath('header')) > 0  and section.xpath('enum') and len(section.xpath('enum'))>0) else
        {
            'section_number': '',
            'section_header': '', 
            'section_text': etree.tostring(section, method="text", encoding="unicode"),
            'section_xml': etree.tostring(section, method="xml", encoding="unicode")
        } 
        for section in sections ]

def xml_to_text(xml_path: str) -> str:
    """
    Parses the xml file and returns the text of the body element, if any
    """
    try:
        billTree = etree.parse(xml_path)
    except:
        raise Exception('Could not parse bill')
    return etree.tostring(billTree, method="text", encoding="unicode")
    """
    Parses the xml file and returns text of the body 
    """
    try:
        billTree = etree.parse(xml_path)
    except:
        raise Exception('Could not parse bill')
    bodies = billTree.xpath('//body')
    if len(sections) == 0:
        return '' 
    return etree.tostring(bodies[0], method="text", encoding="unicode"),

# TODO: Add a function to parse the bill (text) into paragraphs 

# TODO: calculate ngrams from the bill text
