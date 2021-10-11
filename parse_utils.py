# python

from os import path, listdir
from os.path import isfile, join
from types import new_class
from typing import List
from lxml import etree 
import sklearn.feature_extraction.text
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from scipy.spatial.distance import cosine

# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_116_USLM = 'samples/congress/116/uslm'
PATH_116_USLM_TRAIN = 'samples/congress/116/train'
BILLS_SAMPLE = [f'BILLS-116hr{number}ih.xml' for number in range(100, 300)]
BIG_BILLS = ['BILLS-116s1790enr.xml', 'BILLS-116hjres31enr.xml']
BIG_BILLS_PATHS = [path.join(PATH_116_USLM, bill) for bill in (BIG_BILLS + BILLS_SAMPLE)]

SAMPLE_BILL_PATHS_TRAIN = [join(PATH_116_USLM_TRAIN, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM_TRAIN, f))]
SAMPLE_BILL_PATHS = [join(PATH_116_USLM, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM, f))]


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

def text_to_vect(txt: str , ngram_size: int = 4):
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
    # ngrams = vect.get_feature_names_out()
    # print('{1}-grams: {0}'.format(ngrams, ngram_size))
    #print(vect.vocabulary_)
    return vect # list of text documents

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
    bodies = billTree.xpath('//body')
    if len(sections) == 0:
        return '' 
    return etree.tostring(bodies[0], method="text", encoding="unicode"),

def xml_to_vect(xml_paths: List[str], ngram_size: int = 4):
    """
    Parses the xml file and returns the text of the body element, if any
    """
    total_str = '\n'.join([xml_to_text(xml_path) for xml_path in xml_paths])
    return text_to_vect(total_str, ngram_size=ngram_size)

    # to get the vocab dict: vect.vocabulary_

def combine_vocabs(vocabs: List[CountVectorizer]):
    """
    Combines one or more vocabs into one
    """
    vocab_keys = list(set([list(v.vocabulary_.keys()) for v in vocabs]))
    vocab = {vocab_key: str(i) for i, vocab_key in enumerate(vocab_keys)}
    return vocab

def get_combined_vocabs(xml_paths: List[str] = SAMPLE_BILL_PATHS, ngram_size: int = 4):
    """
    Gets the combined vocabulary of all the xml files
    """
    return xml_to_vect(xml_paths, ngram_size=ngram_size)

def getSampleText():
    return xml_to_text(BIG_BILLS_PATHS[0])

def transform_text(text: str, vocab: dict, ngram_size: int = 4):
    """
    Transforms text into a vector using the vocab
    """
    return CountVectorizer(vocabulary=vocab).fit_transform([text])

def train_hashing_vectorizer(train_data: List[str], ngram_size: int = 4):
    """
    Trains a hashing vectorizer on the training data
    """
    vectorizer = HashingVectorizer(ngram_range=(ngram_size,ngram_size))
    X = vectorizer.fit_transform(train_data)
    return vectorizer, X

def test_hashing_vectorizer(vectorizer: HashingVectorizer, test_data: List[str]):
    return vectorizer.transform(test_data)

# TODO: Add a function to parse the bill (text) into paragraphs 

# TODO: calculate ngrams from the bill text
