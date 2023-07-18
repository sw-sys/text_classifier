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

def cpu():
    ### TRANSFORMERS MODEL https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": data,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "device": "cpu"
        }
    )

    ### EXTRACT AND ANALYSE EACH SENTENCE
    # find sentences which represent x% of concepts

    sentence_model = spacy.blank("en")

    # identify sentences - take whole text and identify breaks
    sentence_model.add_pipe("sentencizer")

    with open ("HEALTHLINE_Definitive_Guide_to_Healthy_Eat.txt", "r", encoding='utf-8') as f:
        text = f.read()

    # make object of text
    sentences = sentence_model(text)

    ### ANALYSIS

    # create dict, index = sentence position
    final_data = []

    # iterate over text
    for sentence in sentences.sents:
        doc = nlp(sentence.text)
        final_data.append({"sentence": doc.text, "Categories:": doc._.cats})

    # print by sentence position
    print(final_data[3])

if __name__ == "__main__":
    print("CPU execution time: ", timeit.timeit(cpu, number=1))

### RESULTS OF THIS TIMEIT
# POWER SAVING - CPU execution time:  12.75867839998682
# BALANCED - CPU execution time:  7.491060899978038
# HIGH PERFORMANCE - CPU execution time:  7.5202336000220384

### RESULTS of ARCHIVED TIMEIT
#CPU execution time:  7.68303559999913
#GPU execution time:  7.172171399986837