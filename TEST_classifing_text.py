# Text classifying with spacy and classy classification
# for basic model - run in terminal: python -m spacy download en_core_web_md

import spacy
import classy_classification

# read (#and print) file to terminal
with open("fruits2.txt") as f:
    #print(f.read())

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

# ### BASIC MODEL
# nlp = spacy.load("en_core_web_md")

# # adding in text cat - config tells text cat what to do
# nlp.add_pipe(
#     "text_categorizer",
#     config ={
#         "data": data,
#         "model": "spacy"
#     }
# )

### TRANSFORMERS MODEL https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

nlp = spacy.blank("en")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "device": "gpu" # or cpu but slower
    }
)

# EXECUTE TEXT CATEGORIZER
print(nlp("I like bananas, they are tasty.")._.cats)

### OUTPUT
# basic model output {'fruit': 0.7276752742571169, 'health': 0.2723247257428832}
# TF output {'fruit': 0.7702415759662328, 'health': 0.2297584240337673}