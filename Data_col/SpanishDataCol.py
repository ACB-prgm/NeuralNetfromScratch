from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import pickle
import os


BASE_DIR = os.path.dirname(__file__)
EXPORT_PATH = os.path.join(BASE_DIR, "spanish_nouns.pickle")
URL = "https://eslgrammar.org/list-of-nouns/"
THES = ["el ", "la ", "los ", "las "]
translations = []


def main():
    r = requests.get("https://www.talkenglish.com/vocabulary/top-1500-nouns.aspx").text
    nouns = pd.read_html(r)[3][1]
    print("Translating")
    nouns.apply(translate)

    span_nouns = pd.Series(translations).drop_duplicates()
    span_nouns = span_nouns.drop(span_nouns[span_nouns.str.contains("|".join(THES)) == False].index)

    with open(EXPORT_PATH, "wb") as file:
        pickle.dump(span_nouns, file)


def translate(word):
    r = requests.get(f"https://www.spanishdict.com/translate/{word}").text
    soup = bs(r, features="lxml")
    translation = soup.find_all("div", class_="quickdefWrapper--hSExNXQY")
    for trans in translation:
        translations.append(trans.text)
    
    print(len(translations), end="\r")

    return word



if __name__ == "__main__":
    main()
    print("FINISHED")