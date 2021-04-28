from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd
import emoji
from tqdm import tqdm
import spacy
import torch
import pickle
import config

from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/similarity/{comment_text, sample}")
def process_querie(querie_text: str = None, requested_sample: str = "problems",
                   exist_sample_embedds: bool = False) :
    """
    Load the comment and prepare it for the rest of processing

    Args:
      comment: the comment that gonna be processed
    Return:
      dict: the cleaned dataset
    """
    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT BEFORE PREPROCESSING]:", querie_text, "\n")
    # Clean the text
    text = preprocessing(querie_text)
    # Remove emojis
    text = remove_emoji(text)
    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT AFTER PREPROCESSING]:", text, "\n")
    # Split the text into multiple sub-texts
    sub_texts = split_data(text)
    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT AFTER SPLITTING]: TEXT :{}".format(len(sub_texts)))
    for sent in sub_texts:
        print("\t================================================")
        print("\t" + sent)

    # Loading the samples corpus
    samples_corpus = config.samples[requested_sample]
    print("----------------------------------------------------------------------------------------------------------")
    print("[SAMPLE CORPUS TEXT]:{}".format(len(samples_corpus)))

    # Check if the samples embeddings should be recalculated or just use the saved ones
    embedds_path = config.settings['sample_embedds_path'] + requested_sample + ".pickle"
    if exist_sample_embedds:
        # Loading the embeddings of the samples corpus
        with open(embedds_path, 'rb') as input_file:
            sample_embedds = pickle.load(input_file)
            print("[INFO]: loading exsistant sample mebeddings ...\n")
    else:
        # Creating Embeddings
        sample_embedds = create_embeddings(samples_corpus)
        # Save the calculated embeddings
        with open(embedds_path, 'wb') as output_file:
            pickle.dump(sample_embedds, output_file)

    # Calculate the embeddings of the querie text
    querie_embedds = create_embeddings(sub_texts)
    print("----------------------------------------------------------------------------------------------------------")
    print("[QUERIE EMBEDDINGS]:{}".format(querie_embedds.shape))
    print("[SAMPLES EMBEDDINGS]:{}".format(sample_embedds.shape))

    # Compute similarity
    results = get_similarity(sample_embedds, querie_embedds)
    print("\n----------------------------------------------------------------------------------------------------------")
    print("[RESULTS BEFORE TUNING]:{}".format(results.shape))

    results = tune_results(results, sub_texts)
    print("----------------------------------------------------------------------------------------------------------")
    print("[RESULTS AFTER TUNING]:\n{}".format(results))
    return results


def preprocessing(text: str = None) -> str:
    """ permet détablir la liste des taches suivantes:
      1. Sans contenu html
      2. Sans liens hypertext
      3. Ne garder q'une seule . or ? or !
      4. Sans caractères latins
      5. Sans espaces au début et à la fin de la chaine 3

    Args:
        text: une chaine de charactères
    Return:
        text: la chaine apré le nettoyage
    """
    try:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = re.sub("\\s*'\\B|\\B'\\s*", " ", text)
        text = " ".join(text.split())
        if not text.strip():
            raise ValueError('empty string after removing the preprocessing')
    except ValueError as e:
        print(e)
    return text


def remove_emoji(text: str = None) -> str:
    """ Remove emojis and make sure that the sentence is not empty
    Args:
        text: une chaine de charactères
    Return:
        text: la chaine apré le nettoyage
    """
    try:
        text = emoji.get_emoji_regexp().sub(u'', text)
        if not text.strip():
            raise ValueError('empty string after removing the emojis')
    except ValueError as e:
        print(e)
    return text


def split_data(text: str = None) -> list:
    """
    Split the text using Spacy fr_dep_news_trf that is based on Camembert
    inorder to get shorter comments

    Args:
        text: une chaine de charactères
    Return:
        list[str]: la liste des fragments de text apré le découpage
    """
    split_contents = list()
    comment_idx = list()
    nlp = spacy.load("fr_dep_news_trf", disable=['tagger', 'ner',
                                                 'morphologizer',
                                                 'attribute_ruler', 'lemmatizer'])
    doc = nlp(text)
    sub_sent = list()

    for sentence in doc.sents:
        split_contents.append(sentence.text)
    print(split_contents)

    data = pd.DataFrame({"Doc": split_contents})
    ## Computing word count
    data['word_count'] = data['Doc'].apply(lambda x: len(str(x).split()))
    ## Keeping only reviews with more than one word
    data = data[data["word_count"] > 1]
    data.reset_index(drop=True, inplace=True)
    data.drop('word_count', axis=1, inplace=True)
    return list(data.Doc.values)


def create_embeddings(text_list: list) -> torch.Tensor:
    """
    Create sentences embeddings

    Args:
      comments: list of the sentences
    Return:
      dict: {"sentence":"label"} where label if 0 or 1 (1 = recommendation)
    """
    embedder = SentenceTransformer(config.settings["bert_model"])
    # Query sentences:
    embeddings = embedder.encode(text_list)
    return embeddings


def get_similarity(queries_embedds: np.ndarray,
                   samples_embedds: np.ndarray) -> pd.DataFrame:
    """
    Check if the similartity between the text and all texts inside the
    samples_corpus is greater than or equal to the chosen threshold

    Args:
      samples_embedds: numpy.ndarray array of embeddings of the samples
      queries_embedds: numpy.ndarray array of embeddings of the queries
    Return:
      dict: return dictionary {"text1",(True/False),"text2",(True/False)}
      (similarity >= threshold) or False (similarity <= threshold)
    """
    # Find the closest sentences of the corpus for each query sentence based on cosine similarity
    # Threshold is the lowes similarity to be considered
    Score = pd.DataFrame(cosine_similarity(samples_embedds, queries_embedds))
    print("===>" + str(Score.shape))
    print(Score)
    print(Score.index)

    similarity = list()
    sample_index = list()
    for elem in tqdm(Score.index):
        tmp = Score.loc[elem]
        similarity.append(tmp[tmp.idxmax(axis=1)])
        sample_index.append(tmp.idxmax(axis=1))
        # print("the doc:",elem,"similar to", tmp.idxmax(axis=1),"=",tmp[tmp.idxmax(axis=1)])

    # results.sort_values(by=['similarity'],ascending=False, inplace=True)
    # results.reset_index(drop=True, inplace = True)
    return pd.DataFrame({
        "sentence_idx": Score.index,
        "sample_idx": sample_index,
        "cos_sim": similarity
    })


def tune_results(results: pd.DataFrame, sub_texts: list) -> pd.DataFrame:
    """
    Check if the requested sentences that came from the comment that
    was splited into multiple snetences is a recommendation or not

    Args:
      comments: list of the sentences
      sub_texts: list of sub_text that we got after splitting the text
    Return:
      dict: {"sentence":"label"} where label if 0 or 1 (1 = recommendation)
    """
    results['similarity'] = results.apply(lambda row: row.cos_sim >= 0.70, axis=1)
    results['sentence_text'] = sub_texts
    return results.to_dict(orient="records")

# if __name__ == '__main__':
#
#     querie = "La situation de l'hôtel, la chambre munie d'une petite terrasse avec vue sur le Lot et sur la ville, le diner en terrasse toujours au bord du Lot, c'était vraiment très agréable. Rien"
#     results = process_querie(querie, "recommendation", exist_sample_embedds=True)
#     results.head()
