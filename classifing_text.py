# Text classifying with spacy and classy classification
# d/l model - run in terminal: python -m spacy download en_core_web_md

import spacy
import classy_classification

# read and print file to terminal
with open("fruits2.txt") as f:
    print(f.read())

data = {
    "fruit": ["apple", 
              "Berries such as strawberries, blueberries, and raspberries", 
              "kiwi", 
              "banana",
              "Citrus fruits: Citrus fruits such as oranges, grapefruits, and lemons are rich in vitamin C",
              "When buying and serving fruits, it is important to aim for variety",],

    "health": ["vitamins",
               "minerals",
               "healthy immune system",
               "nutrients",
               "reduce the risk of chronic diseases, improve digestion, promote weight loss, boost immune function, and improve heart health.",],
}

# model
nlp = spacy.load("en_core_web_md")

# adding in text cat - config tells text cat what to do
nlp.add_pipe(
    "text_categorizer",
    config ={
        "data": data,
        "model": "spacy"
    }
)

# execute text cat
print(nlp("I like bananas, they are tasty.")._.cats)