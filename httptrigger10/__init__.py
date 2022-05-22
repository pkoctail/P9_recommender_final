#import libraries
import logging
import azure.functions as func
import os, uuid
import pandas as pd
from azure.storage.blob import BlobClient
import pickle
import pandas as pd
import numpy as np
import recommend as pr
import urllib

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise import KNNWithMeans, SVD
from surprise import accuracy

#download files from blob storage
logging.info('Python HTTP trigger function processed a request.')  

url ='https://p9recommender3.blob.core.windows.net/p9container/articles_metadata.csv'
articles_metadata_df = pd.read_csv(url)

url_1 ='https://p9recommender3.blob.core.windows.net/p9container/df_app.csv'
df_app = pd.read_csv(url_1)
df_app.sort_values('user_id', inplace=True)
df_app.drop_duplicates(subset=['user_id'], inplace=True)
df_app.set_index('user_id', inplace=True, drop=False)

url_2 ='https://p9recommender3.blob.core.windows.net/p9container/model_KNNWithMeans_deploy.pkl'
infile_knn = urllib.request.urlopen(url_2)
model_KNNWithMeans_deploy = pickle.load(infile_knn)

url_3 ='https://p9recommender3.blob.core.windows.net/p9container/articles_embeddings.pickle'
infile_artembedd = urllib.request.urlopen(url_3)
article_embeddings = pickle.load(infile_artembedd)
#article_embeddings_float16 = article_embeddings.astype(np.float16)

#azure functions code triggers http processing, uses functions in recommend library and returns 5 articles
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        articles = pr.getFiveArticles(int(name), model_KNNWithMeans_deploy, articles_metadata_df, df_app, article_embeddings)
        return func.HttpResponse(f"{articles}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )