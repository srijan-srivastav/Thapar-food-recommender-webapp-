
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from time import time


# In[2]:


# import numpy as np

# from lightfm.datasets import fetch_movielens

# movielens = fetch_movielens()


# In[3]:


data = pd.read_csv(r"C:\Users\Sparsh Jain\Downloads\ThaparFoodRecommendation.csv")
restid = pd.read_csv(r"C:\Users\Sparsh Jain\Downloads\RestaurantID.csv")
itemid = pd.read_csv(r"C:\Users\Sparsh Jain\Downloads\ItemID.csv") 
restid_dict = {}
itemid_dict = {} 


# In[4]:


def create_dict(dictionary, csvfile):
    for i in range(len(csvfile)):
        dictionary[csvfile.iloc[i:i+1,0].values[0]] = csvfile.iloc[i:i+1,1].values[0]
    return dictionary


# In[5]:


def fun(id_item, itemid_dict, restid_dict):
    id_rest = itemid_dict[id_item][-1:]
    return(itemid_dict[id_item][:-2] + " From: " + restid_dict[id_rest])


# In[6]:


restid_dict = create_dict(restid_dict, restid)
itemid_dict = create_dict(itemid_dict, itemid)


# In[7]:


restid_dict


# In[8]:


itemid_dict


# In[47]:


fun(3,itemid_dict,restid_dict)


# In[10]:


data.shape


# In[11]:


data


# In[12]:


data = data.iloc[:,1:]


# In[13]:


def create_model(data):
    sparse_matrix = csr_matrix(data.values)
    recommender_obj = LightFM(no_components=30, loss='warp', learning_schedule='adagrad')
    recommender_obj.fit(sparse_matrix, epochs=50)
    return recommender_obj


# In[14]:


model = create_model(data)


# In[142]:


def recommend_item_to_user(model, data, user_id, itemid_dict, threshold = 0, nrec_items = 3, show = False, n_known_likes =5):
    n_items = data.shape[1]

    pred = model.predict(user_id,np.arange(n_items))
    scores = pd.Series(pred)

    # scores.index is from [0,94]
    # data.columns is from [1,95]
    # So, updating indicies of scores  
    scores.index = data.columns

    # "scores" is an object of type Series
    # sorting the key value pair of scores 
    scores = scores.sort_values(ascending=False)

    # saing only the indicies after sorting
    scores = list(scores.index)

    # elements of "scores" are of the type str
    # converting the type to int
    scores = [int(i) for i in scores]

    # retrieving the row of the user with userID = user_id
    userRow = data.iloc[user_id,:]

    # keeping only those elements of the row that have value > 0 (Since threshold = 0)
    userRow = userRow[userRow > threshold]

    # sorting the userRow
    userRowSorted = userRow.sort_values(ascending=False)

    # Now, "userRowSorted" is an obj of type Series
    # saving only the indicies of "userRowSorted"
    userRowSortedIndex = userRowSorted.index

    # elements of "userRowSortedIndex" are of the type str
    # converting the type to int
    userRowSortedIndex = [int(i) for i in userRowSortedIndex]

    # Now, "scores" contains indicies of the scores for all the 95 columns
    # The scores were in descending order
    # "userRowSortedIndex" contains the indicies of all those items which the user has already rated
    # items were sorted from max rating to min rating 

    # Now, we need to extract the scores of those items which the user has not rated yet
    # By doing so we will recommend only those items to the user that the user has not tried yet
    scores = [i for i in scores if i not in userRowSortedIndex]

    # Since "scores" was sorted before it will also be sorted now
    # scores[0] now contains the index of the item that the user has not tried yet 
    # and it is the best item that we can recommend to the user

    # selecting top "nrec_items" items to recommend to the user
    return_score_list = scores[0:nrec_items]

    # known_items will now conatin the names of items that the user has already rated in decreasing order of item rating 
    known_items = []

    for i in userRowSortedIndex:
        temp = fun(i, itemid_dict, restid_dict)
        known_items.append(temp)

    # recommend_items will now contain the names of those items that the user has never tried 
    # and it is most likely that user will like these items
    recommend_items = []

    for i in return_score_list:
        temp = fun(i, itemid_dict, restid_dict)
        recommend_items.append(temp)

    # Printing the Known Likes
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items[0:n_known_likes]:
            print(str(counter) + ": " + i)
            counter+=1
        print("\n")

    # Printing the Recommended Items
    print("Recommended Items:")
    counter = 1
    for i in recommend_items:
        print(str(counter) + ": " + i)
        counter+=1


# In[155]:


recommend_item_to_user(model, data, 10, itemid_dict, show = True)

