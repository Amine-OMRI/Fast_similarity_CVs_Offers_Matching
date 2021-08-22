# Follow these steps to run the project:


* Requirements installation:
    * Run pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3).

* Under Doc_similarity run the following command in the terminal
    * uvicorn main:app --reload

* The application now works on http://127.0.0.1:8000
    * you can check it on http://127.0.0.1:8000/docs#/

# Implemented an API using fastApi, ML and NLP technologies to calculate the similarity between resumes and job postings in order to obtain for each posting the best candidate with the most information matching the job posting.

# Steps : 
* Clean the text
* Split the text into multiple sub-texts
* Creating Embeddings
* Compute similarity
* Tune results
