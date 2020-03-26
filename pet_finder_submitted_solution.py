import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import pprint
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook

np.random.seed(seed=1337)

warnings.filterwarnings('ignore')

split_char = '/'

os.listdir('../input')


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')

import cv2
from keras.applications.densenet import preprocess_input, DenseNet121

def resize_to_square(im):
    old_size = im.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im


def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image

img_size = 256
batch_size = 256


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)

x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


pet_ids = train['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


features = {}

for b in tqdm(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.columns = [f'img_{i}' for i in range(train_feats.shape[1])]


pet_ids = test['PetID'].values
n_batches = len(pet_ids) // batch_size + 1




features = {}

for b in tqdm(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]




test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = [f'img_{i}' for i in range(test_feats.shape[1])]

train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)




all_ids = pd.concat([train, test], axis=0, ignore_index=True, sort=False)[['PetID']]
all_ids.shape




n_components = 32
svd_ = TruncatedSVD(n_components=n_components, random_state=1337)




features_df = pd.concat([train_feats, test_feats], axis=0)
features = features_df[[f'img_{i}' for i in range(256)]].values




svd_col = svd_.fit_transform(features)
svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('IMG_SVD_')
img_features = pd.concat([all_ids, svd_col], axis=1)


import scipy as sp

from functools import partial
from math import sqrt
import matplotlib.pyplot as plt




from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,NMF




from collections import Counter




import lightgbm as lgb

np.random.seed(369)




def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings



def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return (1.0 - numerator / denominator)







class OptimizedRounder(object):

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):

        return self.coef_['x']


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))




train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

print('Breeds')
breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
print(breeds.shape)

print('Colors')
colors = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
print(colors.shape)

print('States')
states = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
print(states.shape)

train_id = train['PetID']
test_id = test['PetID']



doc_sent_mag = []
doc_sent_score = []
first_sentiment=[]
nf_count = 0

for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
            if(sentiment['sentences']):
                first_sentiment.append(sentiment['sentences'][0]['sentiment']['magnitude'])
            else: 
                first_sentiment.append(-1)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
        first_sentiment.append(-1)

train.loc[:, 'doc_sent_mag'] = doc_sent_mag
train.loc[:, 'doc_sent_score'] = doc_sent_score
train.loc[:, 'first_sentiment'] = first_sentiment

doc_sent_mag = []
doc_sent_score = []
first_sentiment=[]
nf_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
            if(sentiment['sentences']):
                first_sentiment.append(sentiment['sentences'][0]['sentiment']['score'])
            else: 
                first_sentiment.append(-1)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
        first_sentiment.append(-1)

test.loc[:, 'doc_sent_mag'] = doc_sent_mag
test.loc[:, 'doc_sent_score'] = doc_sent_score
test.loc[:, 'first_sentiment'] = first_sentiment

train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=2,  max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
# Fit TFIDF

tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)




svd = TruncatedSVD(n_components=60, random_state=1337)
svd.fit(X)
print(svd.explained_variance_ratio_.sum())
print(svd.explained_variance_ratio_)
X2 = svd.transform(X)
X2 = pd.DataFrame(X2, columns=['svd_{}'.format(i) for i in range(60)]) #change range from 120 to 200
train = pd.concat((train, X2), axis=1)

X_test2 = svd.transform(X_test)
X_test2 = pd.DataFrame(X_test2, columns=['svd_{}'.format(i) for i in range(60)])#change range from 120 to 200
test = pd.concat((test, X_test2), axis=1)

train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values


train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

split_char = '/'

#from open kernels 

X = pd.concat([train, test], ignore_index=True, sort=False)
X_temp=X.copy()

from PIL import Image

train_df_ids = train_id
test_df_ids = test_id

train_df_imgs = pd.DataFrame(train_image_files)
train_df_imgs.columns = ['image_filename']
train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])


test_df_imgs = pd.DataFrame(test_image_files)
test_df_imgs.columns = ['image_filename']
test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])


train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)


def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size 

train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])
train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])
train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)


test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x : x[0])
test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x : x[1])
test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)


aggs = {
    'image_size': ['sum','mean', 'var'],
    'width': ['sum','mean', 'var'],
    'height': ['sum','mean', 'var'],
}


agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_train_imgs.columns = new_columns
agg_train_imgs = agg_train_imgs.reset_index()


agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_test_imgs.columns = new_columns
agg_test_imgs = agg_test_imgs.reset_index()


agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
X_temp = X_temp.merge(agg_imgs, how='left', on='PetID')


X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]


X_train_non_null = X_train.fillna(-1)
X_test_non_null = X_test.fillna(-1)


target = X_train_non_null['AdoptionSpeed']


X_train_non_null.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
X_test_non_null.drop(['AdoptionSpeed','PetID'], axis=1, inplace=True)


stat_cols= ['width_mean','height_mean','image_size_mean','width_var','height_var','image_size_var','width_sum','height_sum','image_size_sum']

train=X_train_non_null
test=X_test_non_null


vertex_xst_1 = []
vertex_yst_1 = []
bounding_confidencest_1 = []
bounding_importance_fracst_1 = []
dominant_bluest_1 = []
dominant_greenst_1= []
dominant_redst_1 = []
dominant_blues2t_1 = []
dominant_greens2t_1= []
dominant_pixel_fracst_1 = []
dominant_pixel_fracs2t_1=[]
dominant_scorest_1 = []
dominant_scores2t_1 = []
label_descriptionst_1 = []
label_scorest_1 = []


vertex_xst_2 = []
vertex_yst_2 = []
bounding_confidencest_2 = []
bounding_importance_fracst_2 = []
dominant_bluest_2 = []
dominant_greenst_2= []
dominant_redst_2 = []
dominant_blues2t_2 = []
dominant_greens2t_2= []
dominant_pixel_fracst_2 = []
dominant_pixel_fracs2t_2=[]
dominant_scorest_2 = []
dominant_scores2t_2 = []
label_descriptionst_2 = []
label_scorest_2 = []
nf_count = 0
nl_count = 0

train['PhotoAmt']=train['PhotoAmt'].astype(np.int32)

for pet, nphoto in zip(train_id,train['PhotoAmt']):
    try:
        p=0
        vertex_xs = []
        vertex_ys = []
        bounding_confidences = []
        bounding_importance_fracs = []
        dominant_blues = []
        dominant_greens = []
        dominant_reds = []
        dominant_blues2 = []
        dominant_greens2 = []
        dominant_pixel_fracs = []
        dominant_pixel_fracs2=[]
        dominant_scores = []
        dominant_scores2 = []
        label_descriptions = []
        label_scores = []

        while(p<(nphoto+1)):

            p=p+1

            with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-'+str(p)+'.json', 'r',encoding="UTF-8") as f:

                data = json.load(f)

                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

                vertex_xs.append(vertex_x)

                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

                vertex_ys.append(vertex_y)

                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

                bounding_confidences.append(bounding_confidence)

                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

                bounding_importance_fracs.append(bounding_importance_frac)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('blue')):

                    dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

                else: 

                    dominant_blue=0

                dominant_blues.append(dominant_blue)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('green')):

                    dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

                else: 

                    dominant_green=0

                dominant_greens.append(dominant_green)

                

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('red')):

                    dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

                else: 

                    dominant_red=0

                dominant_reds.append(dominant_red)                

                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

                dominant_pixel_fracs.append(dominant_pixel_frac)

                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

                dominant_scores.append(dominant_score)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']!=1):

                    if(data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color'].get ('blue')):

                        dominant_blue2 = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color']['blue']

                else: 

                    dominant_blue2=0

                dominant_blues2.append(dominant_blue2)

        




                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color'].get ('green')):

                    dominant_green2 = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color']['green']

                else: 

                    dominant_green2=0

                dominant_greens2.append(dominant_green2)        




                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['pixelFraction']

                dominant_pixel_fracs2.append(dominant_pixel_frac)

                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['score']

                dominant_scores2.append(dominant_score)             




                if data.get('labelAnnotations'):

                    label_description = data['labelAnnotations'][0]['description']      

                    label_descriptions.append(label_description)

                    label_score = data['labelAnnotations'][0]['score']

                    label_scores.append(label_score)

                else:

                    nl_count += 1

                    label_descriptions.append('nothing')

                    label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_blues2.append(-1)
        dominant_greens2.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        dominant_pixel_fracs2.append(-1)
        dominant_scores2.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)
     

    if(nphoto):

        vertex_xs_1=np.reshape(vertex_xs,(1,-1))[0][0]
        vertex_ys_1=np.reshape(vertex_ys,(1,-1))[0][0]
        bounding_confidences_1=np.reshape(bounding_confidences,(1,-1))[0][0]
        bounding_importance_fracs_1=np.reshape(bounding_importance_fracs,(1,-1))[0][0]
        dominant_blues_1=np.reshape(dominant_blues,(1,-1))[0][0]
        dominant_greens_1=np.reshape(dominant_greens,(1,-1))[0][0]
        dominant_reds_1=np.reshape(dominant_reds,(1,-1))[0][0]
        dominant_blues2_1=np.reshape(dominant_blues2,(1,-1))[0][0]
        dominant_greens2_1=np.reshape(dominant_greens2,(1,-1))[0][0]
        dominant_pixel_fracs_1=np.reshape(dominant_pixel_fracs,(1,-1))[0][0]
        dominant_scores_1=np.reshape(dominant_scores,(1,-1))[0][0]
        dominant_pixel_fracs2_1=np.reshape(dominant_pixel_fracs2,(1,-1))[0][0]
        dominant_scores2_1=np.reshape(dominant_scores2,(1,-1))[0][0]
        label_scores_1=np.reshape(label_scores,(1,-1))[0][0]
        label_descriptions_1=np.reshape(label_descriptions,(1,-1))[0][0]

    else:

        nf_count += 1

        vertex_xs_1=-1

        vertex_ys_1=-1

        bounding_confidences_1=-1

        bounding_importance_fracs_1=-1

        dominant_blues_1=-1

        dominant_greens_1=-1

        dominant_reds_1=-1

        dominant_blues2_1=-1

        dominant_greens2_1=-1       

        dominant_pixel_fracs_1=-1

        dominant_scores_1=-1

        dominant_pixel_fracs2_1=-1

        dominant_scores2_1=-1

        label_descriptions_1='nothing'

        label_scores_1=-1

    if(len(vertex_xs)>1):

        vertex_xs_2=np.reshape(vertex_xs,(1,-1))[0][1]

        vertex_ys_2=np.reshape(vertex_ys,(1,-1))[0][1]

        bounding_confidences_2=np.reshape(bounding_confidences,(1,-1))[0][1]

        bounding_importance_fracs_2=np.reshape(bounding_importance_fracs,(1,-1))[0][1]

        dominant_blues_2=np.reshape(dominant_blues,(1,-1))[0][1]

        dominant_greens_2=np.reshape(dominant_greens,(1,-1))[0][1]

        dominant_reds_2=np.reshape(dominant_reds,(1,-1))[0][1]

        dominant_blues2_2=np.reshape(dominant_blues2,(1,-1))[0][1]

        dominant_greens2_2=np.reshape(dominant_greens2,(1,-1))[0][1]

        dominant_pixel_fracs_2=np.reshape(dominant_pixel_fracs,(1,-1))[0][1]

        dominant_scores_2=np.reshape(dominant_scores,(1,-1))[0][1]

        dominant_pixel_fracs2_2=np.reshape(dominant_pixel_fracs2,(1,-1))[0][1]

        dominant_scores2_2=np.reshape(dominant_scores2,(1,-1))[0][1]

        label_scores_2=np.reshape(label_scores,(1,-1))[0][1]

        label_descriptions_2=np.reshape(label_descriptions,(1,-1))[0][1]        




    else:

        nf_count += 1

        vertex_xs_2=-1

        vertex_ys_2=-1

        bounding_confidences_2=-1

        bounding_importance_fracs_2=-1

        dominant_blues_2=-1

        dominant_greens_2=-1

        dominant_reds_2=-1

        dominant_blues2_2=-1

        dominant_greens2_2=-1       

        dominant_pixel_fracs_2=-1

        dominant_scores_2=-1

        dominant_pixel_fracs2_2=-1

        dominant_scores2_2=-1

        label_descriptions_2='nothing'

        label_scores_2=-1     

    vertex_xst_1.append(vertex_xs_1)

    vertex_yst_1.append(vertex_ys_1)

    bounding_confidencest_1.append(bounding_confidences_1)

    bounding_importance_fracst_1.append(bounding_importance_fracs_1)

    dominant_bluest_1.append(dominant_blues_1)

    dominant_greenst_1.append(dominant_greens_1)

    dominant_redst_1.append(dominant_reds_1)

    dominant_blues2t_1.append(dominant_blues2_1)

    dominant_greens2t_1.append(dominant_greens2_1)

    dominant_pixel_fracst_1.append(dominant_pixel_fracs_1)

    dominant_pixel_fracs2t_1.append(dominant_pixel_fracs2_1)

    dominant_scorest_1.append(dominant_scores_1)

    dominant_scores2t_1.append(dominant_scores2_1)

    label_descriptionst_1.append(label_descriptions_1)

    label_scorest_1.append(label_scores_1)




    vertex_xst_2.append(vertex_xs_2)

    vertex_yst_2.append(vertex_ys_2)

    bounding_confidencest_2.append(bounding_confidences_2)

    bounding_importance_fracst_2.append(bounding_importance_fracs_2)

    dominant_bluest_2.append(dominant_blues_2)

    dominant_greenst_2.append(dominant_greens_2)

    dominant_redst_2.append(dominant_reds_2)

    dominant_blues2t_2.append(dominant_blues2_2)

    dominant_greens2t_2.append(dominant_greens2_2)

    dominant_pixel_fracst_2.append(dominant_pixel_fracs_2)

    dominant_pixel_fracs2t_2.append(dominant_pixel_fracs2_2)

    dominant_scorest_2.append(dominant_scores_2)

    dominant_scores2t_2.append(dominant_scores2_2)

    label_descriptionst_2.append(label_descriptions_2)

    label_scorest_2.append(label_scores_2)

    

print(nf_count)
print(nl_count)

train.loc[:, 'vertex_x_1'] = vertex_xst_1
train.loc[:, 'vertex_y_1'] = vertex_yst_1
train.loc[:, 'bounding_confidence_1'] = bounding_confidencest_1
train.loc[:, 'bounding_importance_1'] = bounding_importance_fracst_1
train.loc[:, 'dominant_blue_1'] = dominant_bluest_1
train.loc[:, 'dominant_green_1'] = dominant_greenst_1
train.loc[:, 'dominant_red_1'] = dominant_redst_1
train.loc[:, 'dominant_blue2_1'] = dominant_blues2t_1
train.loc[:, 'dominant_green2_1'] = dominant_greens2t_1
train.loc[:, 'dominant_pixel_frac_1'] = dominant_pixel_fracst_1
train.loc[:, 'dominant_score_1'] = dominant_scorest_1
train.loc[:, 'dominant_pixel_frac2_1'] = dominant_pixel_fracs2t_1
train.loc[:, 'dominant_score2_1'] = dominant_scores2t_1
train.loc[:, 'label_description_1'] = label_descriptionst_1
train.loc[:, 'label_score_1'] = label_scorest_1




train.loc[:, 'vertex_x_2'] = vertex_xst_2
train.loc[:, 'vertex_y_2'] = vertex_yst_2
train.loc[:, 'bounding_confidence_2'] = bounding_confidencest_2
train.loc[:, 'bounding_importance_2'] = bounding_importance_fracst_2
train.loc[:, 'dominant_blue_2'] = dominant_bluest_2
train.loc[:, 'dominant_green_2'] = dominant_greenst_2
train.loc[:, 'dominant_red_2'] = dominant_redst_2
train.loc[:, 'dominant_blue2_2'] = dominant_blues2t_2
train.loc[:, 'dominant_green2_2'] = dominant_greens2t_2
train.loc[:, 'dominant_pixel_frac_2'] = dominant_pixel_fracst_2
train.loc[:, 'dominant_score_2'] = dominant_scorest_2
train.loc[:, 'dominant_pixel_frac2_2'] = dominant_pixel_fracs2t_2
train.loc[:, 'dominant_score2_2'] = dominant_scores2t_2
train.loc[:, 'label_description_2'] = label_descriptionst_2
train.loc[:, 'label_score_2'] = label_scorest_2




image_col=['vertex_x_1','vertex_x_2','vertex_y_1','vertex_y_2','dominant_blue_1','dominant_red_1','dominant_green_1','dominant_blue2_1','dominant_green2_1','dominant_blue_2','dominant_red_2','dominant_green_2','dominant_blue2_2','dominant_green2_2','dominant_pixel_frac_1','dominant_pixel_frac2_1','dominant_pixel_frac_2','dominant_pixel_frac2_2','dominant_score_1','dominant_score2_1','dominant_score_2','dominant_score2_2','bounding_confidence_2','bounding_confidence_1','bounding_importance_2','bounding_importance_1','label_score_1','label_score_2']




#image meta data -test 

vertex_xst_1 = []
vertex_yst_1 = []
bounding_confidencest_1 = []
bounding_importance_fracst_1 = []
dominant_bluest_1 = []
dominant_greenst_1= []
dominant_redst_1 = []
dominant_blues2t_1 = []
dominant_greens2t_1= []
dominant_pixel_fracst_1 = []
dominant_pixel_fracs2t_1=[]
dominant_scorest_1 = []
dominant_scores2t_1 = []
label_descriptionst_1 = []
label_scorest_1 = []

vertex_xst_2 = []
vertex_yst_2 = []
bounding_confidencest_2 = []
bounding_importance_fracst_2 = []
dominant_bluest_2 = []
dominant_greenst_2= []
dominant_redst_2 = []
dominant_blues2t_2 = []
dominant_greens2t_2= []
dominant_pixel_fracst_2 = []
dominant_pixel_fracs2t_2=[]
dominant_scorest_2 = []
dominant_scores2t_2 = []
label_descriptionst_2 = []
label_scorest_2 = []
nf_count = 0
nl_count = 0

test['PhotoAmt']=test['PhotoAmt'].astype(np.int32)

for pet, nphoto in zip(test_id,test['PhotoAmt']):
    try:
        p=0
        vertex_xs = []
        vertex_ys = []
        bounding_confidences = []
        bounding_importance_fracs = []
        dominant_blues = []
        dominant_greens = []
        dominant_reds = []
        dominant_blues2 = []
        dominant_greens2 = []
        dominant_pixel_fracs = []
        dominant_pixel_fracs2=[]
        dominant_scores = []
        dominant_scores2 = []
        label_descriptions = []
        label_scores = []

        while(p<(nphoto+1)):

            p=p+1

            with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-'+str(p)+'.json', 'r',encoding="UTF-8") as f:

                data = json.load(f)

                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

                vertex_xs.append(vertex_x)

                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

                vertex_ys.append(vertex_y)

                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

                bounding_confidences.append(bounding_confidence)

                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

                bounding_importance_fracs.append(bounding_importance_frac)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('blue')):

                    dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

                else: 

                    dominant_blue=0

                dominant_blues.append(dominant_blue)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('green')):

                    dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

                else: 

                    dominant_green=0

                dominant_greens.append(dominant_green)

                

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get ('red')):

                    dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

                else: 

                    dominant_red=0

                dominant_reds.append(dominant_red)                

                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

                dominant_pixel_fracs.append(dominant_pixel_frac)

                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

                dominant_scores.append(dominant_score)

                if(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']!=1):

                    if(data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color'].get ('blue')):

                        dominant_blue2 = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color']['blue']

                    else: 

                        dominant_blue2=0

                    if(data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color'].get ('green')):    

                        dominant_green2 = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['color']['green']

                    else:

                        dominant_greens=0

                    dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['pixelFraction']

                    dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][1]['score']

                dominant_blues2.append(dominant_blue2)

                dominant_greens2.append(dominant_green2)        




                dominant_pixel_fracs2.append(dominant_pixel_frac)

                dominant_scores2.append(dominant_score)             




                if data.get('labelAnnotations'):

                    label_description = data['labelAnnotations'][0]['description']      

                    label_descriptions.append(label_description)

                    label_score = data['labelAnnotations'][0]['score']

                    label_scores.append(label_score)

                else:

                    nl_count += 1

                    label_descriptions.append('nothing')

                    label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_blues2.append(-1)

        dominant_greens2.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        dominant_pixel_fracs2.append(-1)

        dominant_scores2.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)

     

    if(nphoto):

        vertex_xs_1=np.reshape(vertex_xs,(1,-1))[0][0]

        vertex_ys_1=np.reshape(vertex_ys,(1,-1))[0][0]

        bounding_confidences_1=np.reshape(bounding_confidences,(1,-1))[0][0]

        bounding_importance_fracs_1=np.reshape(bounding_importance_fracs,(1,-1))[0][0]

        dominant_blues_1=np.reshape(dominant_blues,(1,-1))[0][0]

        dominant_greens_1=np.reshape(dominant_greens,(1,-1))[0][0]

        dominant_reds_1=np.reshape(dominant_reds,(1,-1))[0][0]

        dominant_blues2_1=np.reshape(dominant_blues2,(1,-1))[0][0]

        dominant_greens2_1=np.reshape(dominant_greens2,(1,-1))[0][0]

        dominant_pixel_fracs_1=np.reshape(dominant_pixel_fracs,(1,-1))[0][0]

        dominant_scores_1=np.reshape(dominant_scores,(1,-1))[0][0]

        dominant_pixel_fracs2_1=np.reshape(dominant_pixel_fracs2,(1,-1))[0][0]

        dominant_scores2_1=np.reshape(dominant_scores2,(1,-1))[0][0]

        label_scores_1=np.reshape(label_scores,(1,-1))[0][0]

        label_descriptions_1=np.reshape(label_descriptions,(1,-1))[0][0]

    else:

        nf_count += 1

        vertex_xs_1=-1

        vertex_ys_1=-1

        bounding_confidences_1=-1

        bounding_importance_fracs_1=-1

        dominant_blues_1=-1

        dominant_greens_1=-1

        dominant_reds_1=-1

        dominant_blues2_1=-1

        dominant_greens2_1=-1       

        dominant_pixel_fracs_1=-1

        dominant_scores_1=-1

        dominant_pixel_fracs2_1=-1

        dominant_scores2_1=-1

        label_descriptions_1='nothing'

        label_scores_1=-1

    if(len(vertex_xs)>1):

        vertex_xs_2=np.reshape(vertex_xs,(1,-1))[0][1]

        vertex_ys_2=np.reshape(vertex_ys,(1,-1))[0][1]

        bounding_confidences_2=np.reshape(bounding_confidences,(1,-1))[0][1]

        bounding_importance_fracs_2=np.reshape(bounding_importance_fracs,(1,-1))[0][1]

        dominant_blues_2=np.reshape(dominant_blues,(1,-1))[0][1]

        dominant_greens_2=np.reshape(dominant_greens,(1,-1))[0][1]

        dominant_reds_2=np.reshape(dominant_reds,(1,-1))[0][1]

        dominant_blues2_2=np.reshape(dominant_blues2,(1,-1))[0][1]

        dominant_greens2_2=np.reshape(dominant_greens2,(1,-1))[0][1]

        dominant_pixel_fracs_2=np.reshape(dominant_pixel_fracs,(1,-1))[0][1]

        dominant_scores_2=np.reshape(dominant_scores,(1,-1))[0][1]

        dominant_pixel_fracs2_2=np.reshape(dominant_pixel_fracs2,(1,-1))[0][1]

        dominant_scores2_2=np.reshape(dominant_scores2,(1,-1))[0][1]

        label_scores_2=np.reshape(label_scores,(1,-1))[0][1]

        label_descriptions_2=np.reshape(label_descriptions,(1,-1))[0][1]        




    else:

        nf_count += 1

        vertex_xs_2=-1

        vertex_ys_2=-1

        bounding_confidences_2=-1

        bounding_importance_fracs_2=-1

        dominant_blues_2=-1

        dominant_greens_2=-1

        dominant_reds_2=-1

        dominant_blues2_2=-1

        dominant_greens2_2=-1       

        dominant_pixel_fracs_2=-1

        dominant_scores_2=-1

        dominant_pixel_fracs2_2=-1

        dominant_scores2_2=-1

        label_descriptions_2='nothing'

        label_scores_2=-1     

    vertex_xst_1.append(vertex_xs_1)

    vertex_yst_1.append(vertex_ys_1)

    bounding_confidencest_1.append(bounding_confidences_1)

    bounding_importance_fracst_1.append(bounding_importance_fracs_1)

    dominant_bluest_1.append(dominant_blues_1)

    dominant_greenst_1.append(dominant_greens_1)

    dominant_redst_1.append(dominant_reds_1)

    dominant_blues2t_1.append(dominant_blues2_1)

    dominant_greens2t_1.append(dominant_greens2_1)

    dominant_pixel_fracst_1.append(dominant_pixel_fracs_1)

    dominant_pixel_fracs2t_1.append(dominant_pixel_fracs2_1)

    dominant_scorest_1.append(dominant_scores_1)

    dominant_scores2t_1.append(dominant_scores2_1)

    label_descriptionst_1.append(label_descriptions_1)

    label_scorest_1.append(label_scores_1)




    vertex_xst_2.append(vertex_xs_2)

    vertex_yst_2.append(vertex_ys_2)

    bounding_confidencest_2.append(bounding_confidences_2)

    bounding_importance_fracst_2.append(bounding_importance_fracs_2)

    dominant_bluest_2.append(dominant_blues_2)

    dominant_greenst_2.append(dominant_greens_2)

    dominant_redst_2.append(dominant_reds_2)

    dominant_blues2t_2.append(dominant_blues2_2)

    dominant_greens2t_2.append(dominant_greens2_2)

    dominant_pixel_fracst_2.append(dominant_pixel_fracs_2)

    dominant_pixel_fracs2t_2.append(dominant_pixel_fracs2_2)

    dominant_scorest_2.append(dominant_scores_2)

    dominant_scores2t_2.append(dominant_scores2_2)

    label_descriptionst_2.append(label_descriptions_2)

    label_scorest_2.append(label_scores_2)

    
print(nf_count)
print(nl_count)

test.loc[:, 'vertex_x_1'] = vertex_xst_1
test.loc[:, 'vertex_y_1'] = vertex_yst_1
test.loc[:, 'bounding_confidence_1'] = bounding_confidencest_1
test.loc[:, 'bounding_importance_1'] = bounding_importance_fracst_1
test.loc[:, 'dominant_blue_1'] = dominant_bluest_1
test.loc[:, 'dominant_green_1'] = dominant_greenst_1
test.loc[:, 'dominant_red_1'] = dominant_redst_1
test.loc[:, 'dominant_blue2_1'] = dominant_blues2t_1
test.loc[:, 'dominant_green2_1'] = dominant_greens2t_1
test.loc[:, 'dominant_pixel_frac_1'] = dominant_pixel_fracst_1
test.loc[:, 'dominant_score_1'] = dominant_scorest_1
test.loc[:, 'dominant_pixel_frac2_1'] = dominant_pixel_fracs2t_1
test.loc[:, 'dominant_score2_1'] = dominant_scores2t_1
test.loc[:, 'label_description_1'] = label_descriptionst_1
test.loc[:, 'label_score_1'] = label_scorest_1

test.loc[:, 'vertex_x_2'] = vertex_xst_2
test.loc[:, 'vertex_y_2'] = vertex_yst_2
test.loc[:, 'bounding_confidence_2'] = bounding_confidencest_2
test.loc[:, 'bounding_importance_2'] = bounding_importance_fracst_2
test.loc[:, 'dominant_blue_2'] = dominant_bluest_2
test.loc[:, 'dominant_green_2'] = dominant_greenst_2
test.loc[:, 'dominant_red_2'] = dominant_redst_2
test.loc[:, 'dominant_blue2_2'] = dominant_blues2t_2
test.loc[:, 'dominant_green2_2'] = dominant_greens2t_2
test.loc[:, 'dominant_pixel_frac_2'] = dominant_pixel_fracst_2
test.loc[:, 'dominant_score_2'] = dominant_scorest_2
test.loc[:, 'dominant_pixel_frac2_2'] = dominant_pixel_fracs2t_2
test.loc[:, 'dominant_score2_2'] = dominant_scores2t_2
test.loc[:, 'label_description_2'] = label_descriptionst_2
test.loc[:, 'label_score_2'] = label_scorest_2




rescuers=Counter(train['RescuerID'])
df_rescuers = pd.DataFrame.from_dict(rescuers, orient='index').reset_index()
rescuers_dict = df_rescuers.set_index('index').T.to_dict('records')[0]
Animals_per_rescuer=pd.DataFrame()

for s in train.index:
    Animals_per_rescuer.loc[s,0]=-rescuers_dict[train.loc[s,'RescuerID']]

train['Animals_per_rescuer']=Animals_per_rescuer.rank()/len(train)

#test rescuer

rescuers=Counter(test['RescuerID'])
df_rescuers = pd.DataFrame.from_dict(rescuers, orient='index').reset_index()
rescuers_dict = df_rescuers.set_index('index').T.to_dict('records')[0]
Animals_per_rescuer=pd.DataFrame()

for s in test.index:
    Animals_per_rescuer.loc[s,0]=-rescuers_dict[test.loc[s,'RescuerID']]

test['Animals_per_rescuer']=Animals_per_rescuer.rank()/len(test)


train.drop(['Name', 
            'RescuerID',
            'label_description_1',
            'label_description_2',
            'Description'], axis=1, inplace=True)

test.drop(['Name', 
           'RescuerID',
           'label_description_1',
           'label_description_2',
           'Description'], axis=1, inplace=True)

numeric_cols = ['first_sentiment','Animals_per_rescuer','Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt','label_score','max_label_score', 'doc_sent_mag', 'doc_sent_score', 'dominant_score','max_dominant_scores','dominant_score2', 'dominant_pixel_frac','dominant_pixel_frac2', 'dominant_red','dominant_green','max_dominant_greens','dominant_green2', 'dominant_blue','max_dominant_blues','dominant_blue2', 'bounding_importance', 'bounding_confidence', 'vertex_x', 'vertex_y'] + ['svd_{}'.format(i) for i in range(80)]+image_col+stat_cols

##customized feature

trainxx=train[["Vaccinated","Dewormed","Sterilized"]]
trainxx=trainxx.astype("int32")
r1=trainxx["Vaccinated"].mul(trainxx["Dewormed"])
r2=r1.mul(trainxx["Sterilized"])   
train['H_uncertainty']=r2


Dog_Photo_Breed=(((train['Breed1']==307))*(train['PhotoAmt']))*train['Breed1']


train['Sterilized_Breed_cat']=(((train['Breed1']==264))*(train['Sterilized']))*train['Breed1']+(((train['Breed1']==265))*(train['Sterilized']))*train['Breed1']+(((train['Breed1']==266))*(train['Sterilized']))*train['Breed1']


train['Vacc_Breed_cat']=(((train['Breed1']==264))*(train['Vaccinated']))*train['Breed1']+(((train['Breed1']==265))*(train['Vaccinated']))*train['Breed1']+(((train['Breed1']==266))*(train['Vaccinated']))*train['Breed1']


train['Cat_Photo_Breed']=(((train['Breed1']==266))*(train['PhotoAmt']))*train['Breed1']+(((train['Breed1']==265))*(train['PhotoAmt']))*train['Breed1']+(((train['Breed1']==264))*(train['PhotoAmt']))*train['Breed1']
train['Cat_Photo_Breed']=307*(train['Breed1']==307)+train['Cat_Photo_Breed']
train['Cat_State_Breed']=(((train['Breed1']==266))*(train['State']-41000))*train['Breed1']+(((train['Breed1']==265))*(train['State']-41000))*train['Breed1']+(((train['Breed1']==266))*(train['State']-41000))*train['Breed1']

testxx=test[["Vaccinated","Dewormed","Sterilized"]]
testxx=testxx.astype("int32")
r1=testxx["Vaccinated"].mul(testxx["Dewormed"])
r2=r1.mul(testxx["Sterilized"])   
test['H_uncertainty']=r2




train['Dog_H_breed']=(train['Breed1']==307)*train['H_uncertainty']+(np.logical_or(np.logical_or(train['Breed1']==266,train['Breed1']==265), train['Breed1']==264))*train['H_uncertainty']




Dog_Photo_Breed=(((test['Breed1']==307))*(test['PhotoAmt']))*test['Breed1']




test['Sterilized_Breed_cat']=((test['Breed1']==264)*test['Sterilized'])*test['Breed1']+((test['Breed1']==265)*test['Sterilized'])*test['Breed1']+((test['Breed1']==266)*test['Sterilized'])*test['Breed1']


test['Vacc_Breed_cat']=(((test['Breed1']==264))*(test['Vaccinated']))*test['Breed1']+(((test['Breed1']==265))*(test['Vaccinated']))*test['Breed1']+(((test['Breed1']==266))*(test['Vaccinated']))*test['Breed1']


test['Cat_Photo_Breed']=(((test['Breed1']==266))*(test['PhotoAmt']))*test['Breed1']+(((test['Breed1']==265))*(test['PhotoAmt']))*test['Breed1']+(((test['Breed1']==264))*(test['PhotoAmt']))*test['Breed1']
test['Cat_Photo_Breed']=307*(test['Breed1']==307)+test['Cat_Photo_Breed']


test['Cat_State_Breed']=(((test['Breed1']==266))*(test['State']-41000))*test['Breed1']+(((test['Breed1']==265))*(test['State']-41000))*test['Breed1']+(((test['Breed1']==264))*(test['State']-41000))*test['Breed1']

test['Dog_H_breed']=(test['Breed1']==307)*test['H_uncertainty']+(np.logical_or(np.logical_or(test['Breed1']==266,test['Breed1']==265), test['Breed1']==264))*test['H_uncertainty']

cat_cols = list(set(train.columns) - set(numeric_cols))

train.loc[:, cat_cols] = train[cat_cols].astype('category')
test.loc[:, cat_cols] = test[cat_cols].astype('category')


train['first_sentiment']=train['first_sentiment'].astype(np.float32)
test['first_sentiment']=test['first_sentiment'].astype(np.float32)

train['blue_diff']=np.square(train['dominant_blue_1'].subtract(train['dominant_blue2_1']))
train['green_diff']=np.square(train['dominant_green_1'].subtract(train['dominant_green2_1']))
test['blue_diff']=np.square(test['dominant_blue_1'].subtract(test['dominant_blue2_1']))
test['green_diff']=np.square(test['dominant_green_1'].subtract(test['dominant_green2_1']))


train['toppixels']=train['dominant_pixel_frac_1']+train['dominant_pixel_frac2_1']
test['toppixels']=test['dominant_pixel_frac_1']+test['dominant_pixel_frac2_1']

train['toppixels_2']=train['dominant_pixel_frac_2']+train['dominant_pixel_frac2_2']
test['toppixels_2']=test['dominant_pixel_frac_2']+test['dominant_pixel_frac2_2']




train['gap_senti']=train['first_sentiment']-train['doc_sent_mag']
test['gap_senti']=test['first_sentiment']-test['doc_sent_mag']

#from wikipedia 

state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# state population: https://en.wikipedia.org/wiki/Malaysia

state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}


train["state_gdp"] = train.State.map(state_gdp)
train["state_population"] = train.State.map(state_population)

test["state_gdp"] = test.State.map(state_gdp)
test["state_population"] = test.State.map(state_population)

train_feats.drop(['PetID'],axis=1, inplace=True)
test_feats.drop(['PetID'],axis=1, inplace=True)


train = pd.concat((train, train_feats), axis=1)

test=test.reset_index()
test.drop(['index'],axis=1,inplace=True)
test = pd.concat((test, test_feats), axis=1)


train.shape
test.shape


trainx=pd.concat([train,target],axis=1)

train_0= trainx.loc[target==0]
train_1= trainx.loc[target==1]
train_2= trainx.loc[target==2]
train_3= trainx.loc[target==3]
train_4= trainx.loc[target==4]

nsam=len(train_0)

trainsub1=train_1.sample(n=nsam,random_state=1)
trainsub2=train_2.sample(n=nsam,random_state=1)
trainsub3=train_3.sample(n=nsam, random_state=1)
trainsub4=train_4.sample(n=nsam, random_state=1)

dfs = [train_0,trainsub1, trainsub2, trainsub3,trainsub4]
myvalidation= pd.concat(dfs, ignore_index=True)
mytarget=myvalidation['AdoptionSpeed']
myvalidation.drop(['AdoptionSpeed'],axis=1,inplace=True)


class OptimizedRounder(object):

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds

    def coefficients(self):
        return self.coef_['x']
        
def run_cv_model2(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], 5))
    all_coefficients = np.zeros((5, 4))
    feature_importance_df = pd.DataFrame()
    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/8')

        if isinstance(train, pd.DataFrame):

            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]

        else:

            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()
        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients

        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = train.columns.values
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        

        i += 1

    print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}

    return results
    
def runLGB(train_X, train_y, test_X, test_y, test_X2, params):

    print('Prep LGB')

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]

    print('Train LGB')

    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None

    if params.get('early_stop'):
        early_stop = params.pop('early_stop')

    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()

    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk

params2 = {'application': 'regression', #'regression' 'multiclass'
          'boosting': 'gbdt', # 'rf' 'dart' 'goss'
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'lambda_l2': 0.0475,#0.0475
#          'bagging_freq' :10,
          'learning_rate': 0.015, #0.015
#          'bagging_fraction': 0.95,
          'feature_fraction': 0.95,
          'min_split_gain': 0.01,
          'min_child_samples': 200,
          'min_child_weight': 0.01,
          'verbosity': -1,
          'data_random_seed': 17,
          'early_stop': 350,
          'verbose_eval': 1000,
          'num_rounds': 15000}

trainx=train.copy()
testx=test.copy()


#trainx['from3']=(trainx['Age']-3)^2
#testx['from3']=(testx['Age']-3)^2
trainx['from4']=(trainx['Age']-4)^2
testx['from4']=(testx['Age']-4)^2
trainx['from5']=(trainx['Age']-5)^2
testx['from5']=(testx['Age']-5)^2
trainx['from6']=(trainx['Age']-6)^2
testx['from6']=(trainx['Age']-6)^2
trainx['from7']=(trainx['Age']-7)^2
testx['from7']=(trainx['Age']-7)^2
trainx['isadult']=(trainx['Age']>12)*trainx['Age']/12
testx['isadult']=(testx['Age']>12)*testx['Age']/12

results3 = run_cv_model2(trainx, testx, target, runLGB, params2, rmse, 'lgb')

imports = results3['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imports.sort_values('importance', ascending=False)


optR = OptimizedRounder()
coefficients_ = np.mean(results3['coefficients'], axis=0)
print(coefficients_)
coefficients_[0] = 1.5
coefficients_[1] = 2.135
coefficients_[2] = 2.61113466
coefficients_[3] = 2.94
test_predictions = [r[0] for r in results3['test']]
test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
Counter(test_predictions)
#17
#pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.head()
#19
submission.to_csv('submission.csv', index=False)