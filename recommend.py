
# # Project 9
# Functions that are used to return 5 articles given the required input. 
#The last function ensures that all userIDs in the dataset can used for recommendations

#import libraries
import numpy as np 
import pandas as pd 
import os
import pickle 
from scipy.spatial import distance


#CF function: this function takes the userid, model pickle file and the metadata article df and returns 5 article recommendations
def getFiveArticles_collaborative_filtered(user_id, model_KNNWithMeans_deploy, articles_metadata_df):
    predictions = {}
    #Categories are from 1 to 460
    for i in range(1, 460):
        _, cat_id, _, est, err = model_KNNWithMeans_deploy.predict(user_id, i)
        #Keep prediction only if no error.
        if (err != True):
            predictions[cat_id] = est
    best_cats_to_recommend = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
    recommended_articles = []
    for key, _ in best_cats_to_recommend.items():
        recommended_articles.append(int(articles_metadata_df[articles_metadata_df['category_id'] == key]['article_id'].sample(1).values))
    return recommended_articles

#CB function: this function takes df that includes list of users, article embeddings, user ID and returns 5 articles
def getFiveArticles_content_based(df_app, article_embeddings, userId):
    ee=article_embeddings
    #get all articles read by user
    var= df_app.loc[df_app['user_id']==userId]['article_id'].tolist()
    #chose last in list since this most recent
    value = var[-1]
    #delete all read articles except the selected one( we do not want to offer user to read something he already read)
    for i in range(0, len(var)):
        if i != value:
            ee=np.delete(ee,[i],0)
    arr=[]
    #delete selected article in the new matrix
    f=np.delete(ee,[value],0)
    #get 5 articles the most similar to the selected one
    for i in range(0,5):
        distances = distance.cdist([ee[value]], f, "cosine")[0]
        min_index = np.argmin(distances)
        f=np.delete(f,[min_index],0)
        #find corresponding matrix in original martix
        result = np.where(article_embeddings == f[min_index])
        arr.append(result[0][0])
    return arr

def getFiveArticles(userId, model_KNNWithMeans_deploy_arg, articles_metadata_df_arg, df_app_arg, article_embeddings_arg) :
    if userId in df_app_arg.index :
        articles = getFiveArticles_collaborative_filtered(userId, model_KNNWithMeans_deploy_arg, articles_metadata_df_arg) 
    else : 
        articles=getFiveArticles_content_based(userId, df_app_arg, article_embeddings_arg)
    
    return articles

def main():
    pass 

if __name__ == "__main__":
    main()
