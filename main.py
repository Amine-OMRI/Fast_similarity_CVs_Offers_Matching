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
import time
from fastapi import FastAPI, HTTPException

import xgboost as xgb

# initialise the app
app = FastAPI()
# load the nlp model that gonna be used to split the input text
nlp = spacy.load("fr_dep_news_trf", disable=['tagger', 'ner',
                                             'morphologizer',
                                             'attribute_ruler', 'lemmatizer'])
# load the transformer model that will be used to create embeddings
embedder = SentenceTransformer(config.settings["bert_model"])

@app.get("/api/similarity/{querie_text, requested_sample, exist_sample_embedds}")
async def process_querie(querie_text: str = None, requested_sample: str = "recommendations",
                   exist_sample_embedds: bool = True) -> list :
    """
    Load the comment and prepare it for the rest of processing

    Args:
      querie_text: the comment that gonna be processed
      requested_sample: the list of samples that is stored in config.py (the default one is recommendations)
    Return:
      dict: the cleaned dataset
    """
    start_time = time.time()
    # Exceptions hundling
    if requested_sample not in config.samples.keys():
        raise HTTPException(
            status_code=404,
            detail="requested sample not found",
            headers={"X-Error": "There goes an error"},
        )
    results = await compute_similarity(querie_text, requested_sample, exist_sample_embedds)
    print("--- Global time %s seconds ---" % (time.time() - start_time))

    return results

async def compute_similarity(querie_text: str, requested_sample: str, exist_sample_embedds: bool)-> list:

    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT BEFORE PREPROCESSING]:", querie_text, "\n")
    # Clean the text

    start_time = time.time()
    text = clean_text(querie_text)
    print("--- Cleaning time time %s seconds ---" % (time.time() - start_time))

    # Remove emojis
    # start_time = time.time()
    # text = remove_emoji(text)
    # print("--- Removing Emojis time %s seconds ---" % (time.time() - start_time))

    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT AFTER PREPROCESSING]:", text, "\n")
    # Split the text into multiple sub-texts
    start_time = time.time()
    sub_texts = split_data(nlp, text)
    print("--- Text Splitting time %s seconds ---" % (time.time() - start_time))

    print("----------------------------------------------------------------------------------------------------------")
    print("[TEXT AFTER SPLITTING]: TEXT :{}".format(len(sub_texts)))
    # for sent in sub_texts:
    #     print("\t================================================")
    #     print("\t" + sent)

    # Loading the samples corpus
    samples_corpus = config.samples[requested_sample]
    print("----------------------------------------------------------------------------------------------------------")
    print("[SAMPLE CORPUS TEXT]:{}".format(len(samples_corpus)))

    # Check if the samples embeddings should be recalculated or just use the saved ones
    embedds_path = config.settings['sample_embedds_path'] + requested_sample + ".pickle"
    if exist_sample_embedds:
        # Loading the embeddings of the samples corpus
        start_time = time.time()
        with open(embedds_path, 'rb') as input_file:
            sample_embedds = pickle.load(input_file)
            print("[INFO]: loading Exsistant Sample Embeddings ...\n")
        print("--- Loading Sample Embeddings time %s seconds ---" % (time.time() - start_time))
    else:
        # Creating Embeddings
        start_time = time.time()
        sample_embedds = create_embeddings(embedder, samples_corpus)
        print("--- Calculating Sample Embeddings time %s seconds ---" % (time.time() - start_time))

        # Save the calculated embeddings
        with open(embedds_path, 'wb') as output_file:
            pickle.dump(sample_embedds, output_file)

    # Calculate the embeddings of the querie text
    start_time = time.time()
    querie_embedds = create_embeddings(embedder, sub_texts)
    print("--- Calculating Querie Embeddings time %s seconds ---" % (time.time() - start_time))

    print("----------------------------------------------------------------------------------------------------------")
    print("[QUERIE EMBEDDINGS]:{}".format(querie_embedds.shape))
    print("[SAMPLES EMBEDDINGS]:{}".format(sample_embedds.shape))

    # Compute similarity
    start_time = time.time()
    results = get_similarity(sample_embedds, querie_embedds)
    print("--- Getting Similarity time %s seconds ---" % (time.time() - start_time))

    print("\n----------------------------------------------------------------------------------------------------------")
    print("[RESULTS BEFORE TUNING]:{}".format(results.shape))

    # Tune results
    start_time = time.time()
    results = tune_results(results, sub_texts)
    print("--- Tuning Results time %s seconds ---" % (time.time() - start_time))

    print("----------------------------------------------------------------------------------------------------------")
    if (results.similarity == True).any():
        start_time = time.time()
        results = results[results.similarity == True]
        # Sentiment analysis
        results = classify_results(results, querie_embedds)
        print("--- Classification Results time %s seconds ---" % (time.time() - start_time))
        print("----------------------------------------------------------------------------------------------------------")
        print("[RESULTS AFTER CLASSIFICATION]:\n{}".format(results.shape))

    return results.to_dict(orient="records")

def clean_text(text: str = None) -> str:
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

def split_data(nlp, text: str = None) -> list:
    """
    Split the text using Spacy fr_dep_news_trf that is based on Camembert
    inorder to get shorter comments

    Args:
        text: une chaine de charactères
    Return:
        list[str]: la liste des fragments de text apré le découpage
    """
    split_contents = list()
    doc = nlp(text)

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

def create_embeddings(embedder, text_list: list) -> torch.Tensor:
    """
    Create sentences embeddings

    Args:
      comments: list of the sentences
    Return:
      dict: {"sentence":"label"} where label if 0 or 1 (1 = recommendation)
    """
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
    score = pd.DataFrame(cosine_similarity(samples_embedds, queries_embedds))
    # print("===>" + str(score.shape))
    # print(score)
    # print(score.index)

    similarity = list()
    sample_index = list()
    for elem in tqdm(score.index):
        tmp = score.loc[elem]
        similarity.append(tmp[tmp.idxmax(axis=1)])
        sample_index.append(tmp.idxmax(axis=1))
        # print("the doc:",elem,"similar to", tmp.idxmax(axis=1),"=",tmp[tmp.idxmax(axis=1)])

    # results.sort_values(by=['similarity'],ascending=False, inplace=True)
    # results.reset_index(drop=True, inplace = True)
    return pd.DataFrame({
        "sentence_idx": score.index,
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
    threshold = config.settings["threshold"]
    # print(results['cos_sim'])
    results['similarity'] = results.apply(lambda row: row.cos_sim >=threshold, axis=1)
    results['sentence_text'] = sub_texts
    print(results['sentence_text'])
    print(results['similarity'])

    # results = results[results.similarity == True]
    return results

def classify_results(results: pd.DataFrame, queries_embedds: np.ndarray, xgb_model: str = 'model300K.bin') -> pd.DataFrame:
    """
    Check if the requested sentences that came from the comment that
    was splited into multiple snetences is a recommendation or not

    Args:
      results: pd.DataFrame the output of recommendation detection pipeline
      queries_embedds: numpy.ndarray array of embeddings of the queries
      xgb_model: str the name of the pretrained XGBoost model
    Return:
      dict: {"sentence":"label"} where label if 0 or 1 (1 = recommendation)
    """
    model = xgb.Booster()
    #print(f'##############{config.settings["trained_xgb_model_path"]}{xgb_model}')
    model.load_model(f'{config.settings["trained_xgb_model_path"]}{xgb_model}')
    results_embeddings = np.array([queries_embedds[idx] for idx in results.sentence_idx])
    d_results_embeddings = xgb.DMatrix(results_embeddings)
    comments_predict = model.predict(d_results_embeddings)
    # predicting
    predictions = [round(value) for value in comments_predict]
    results = pd.DataFrame({
        "sentence": results.sentence_text,
        "recommendation": predictions})
    return results


# if __name__ == '__main__':
#
#     querie = "La situation de l'hôtel, la chambre munie d'une petite terrasse avec vue sur le Lot et sur la ville, le diner en terrasse toujours au bord du Lot, c'était vraiment très agréable. Rien"
#     results = process_querie(querie, "recommendation", exist_sample_embedds=True)
#     results.head()

