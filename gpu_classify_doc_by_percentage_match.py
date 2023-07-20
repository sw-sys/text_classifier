# Comparing how much faster classification is with gpu vs cpu
import timeit

### CLASSIFIER TEST
# Finding the occurance of fruit in a healthy eating guide
# with spacy and classy classification

import spacy
import classy_classification

# fruit dictionary created from
# https://simple.wikipedia.org/wiki/List_of_fruits
# health dictionary - sources various

data = {
        "fruit": ["Native British fruits such as apple and pear", 
              "Berries such as acai, strawberries, blackberry, cloudberry, blueberries, cherries, currents, gooseberries, red current, goji berry and raspberries", 
              "Exotic fruits like mango, dragonfruit, pineapple, kiwi, banana, starfruit, dates, figs, jujube, lychee, jackfruit",
              "Melons like cantaloupe, honeydew, musk, watermelon, galia",
              "grapes",
              "Stone fruits such as apricot, plum, nectarine and peach"
              "Citrus fruits: Citrus fruits such as oranges, pomelo, grapefruits, limes and lemons are rich in vitamin C",
              "When buying and serving fruits, it is important to aim for variety",],

        "food": ["protein and carbs",
                 "saturated and unsaturated fat"
                 "sugar",
                 "fruit",
                 "veg",
                 "fish",
                 "nuts",
                 "legumes",
                 "chocolate",
                 "crisps",
                 "macros"
        ],

        "health": ["high in vitamins and minerals",
               "fitness",
               "energy and wellness",
               "survival",
               "wellbeing",
               "quality sleep",
               "strong immune system",
               "nutrients",
               "lower risk of disease",
               "lower mortality rate",
               "reduce the risk of chronic diseases, improve digestion, promote weight loss, boost immune function, and improve heart health.",],
}

def gpu():
    ### TRANSFORMERS MODEL https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": data,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "device": "gpu"
        }
    )

    ### EXTRACT AND ANALYSE EACH SENTENCE
    # find sentences which represent x% of concepts

    sentence_model = spacy.blank("en")

    # identify sentences - take whole text and identify breaks
    sentence_model.add_pipe("sentencizer")

    with open ("chem_food_nutrition.txt", "r", encoding='utf-8') as f:
        text = f.read()

    # make object of text
    sentences = sentence_model(text)

    ### CATEGORISATION

    # create dict, index = sentence position
    final_data = []

    # iterate over text
    for sentence in sentences.sents:
        doc = nlp(sentence.text)
        final_data.append({"sentence": doc.text, "cats": doc._.cats})

    ### ANALYSIS

    # for item in final_data[:2]:
    #     print(item["sentence"])
    #     print(item["cats"]["fear"])

    for i, item in enumerate(final_data[1:1000]):
        if item["cats"]["fruit"] > .70:
            print(f"Item {i+1}:")
            print(item["sentence"])
            print(item["cats"])

### EXECUTE TIME IT
if __name__ == "__main__":
    print("GPU execution time: ",timeit.timeit(gpu, number=1))

### RESULTS OF TEXTBOOK TIME IT
# BALANCED - GPU execution time: 7.556677999993553

### RESULTS of SMALL TEXT TIME IT 
# POWER SAVING - GPU execution time:  12.33139380000648
# BALANCED - GPU execution time:  7.556504999985918
# HIGH PERFORMANCE - GPU execution time:  7.656372500001453

### RESULTS of ARCHIVED TIMEIT
#CPU execution time:  7.68303559999913
#GPU execution time:  7.172171399986837