import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from nltk.corpus import stopwords

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# Some data structures that will be useful
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# Generate a negative set

userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))

lUserSet = list(userSet)
lBookSet = list(bookSet)

notRead = set()
# For each (user,book) entry in the validation set, sample a
# negative entry by randomly choosing a book that user hasn’t read
for u,b,r in ratingsValid:
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    maxItemSim = 0
    avgItemRating = 0
    books = set(ratingsPerUser[u])
    # Jaccard item-to-item similarity
    for u2,r in ratingsPerItem[b]:
        sim = Jaccard(books, set(ratingsPerUser[u2]))
        if sim > maxItemSim:
            maxItemSim = sim
        avgItemRating += r
    avgItemRating = 0 if len(ratingsPerItem[b]) == 0 else avgItemRating/len(ratingsPerItem[b])
    
    maxUserSim = 0
    avgUserRating = 0
    users = set(ratingsPerItem[b])
    # Jaccard user-to-user similarity
    for b2,r in ratingsPerUser[u]:
        sim = Jaccard(users,set(ratingsPerItem[b2]))
        if sim > maxUserSim:
            maxUserSim = sim
            avgUserRating += r
    avgUserRating = 0 if len(ratingsPerUser[u]) == 0 else avgUserRating/len(ratingsPerUser[u])
    
    pred = 0
    if maxUserSim > 0.02 or maxItemSim > 0.2 or len(ratingsPerItem[b]) > 35:
        pred = 1
    # if avgItemRating is above a threshold 
    # and the item has a sufficient number of ratings to source a reliable average
    # if avgItemRating > 3.5 and len(ratingsPerItem[b]) > 20:
        # pred = 1
    # if avgUserRating is within range [avgItemRating-.3, avgItemRating+.3]
    # if avgItemRating-.3 <= avgUserRating <= avgItemRating+.3:
        # pred = 1
        
    _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

predictions.close()



##################################################
# Category prediction (CSE158 only)              #
##################################################
data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)

genresPerUser = {}
# avgGenresRatingPerUser = {}
# globalGenreAvg = [0, 0, 0, 0, 0]
# totalReviews = [0, 0, 0, 0, 0]
for d in data:
    u = d['user_id']
    gid = d['genreID']
    if u not in genresPerUser:
        genresPerUser[u] = [0]*5
#     if u not in avgGenresRatingPerUser:
#         avgGenresRatingPerUser[u] = [0]*5
    genresPerUser[u][gid] += 1
#     avgGenresRatingPerUser[u][gid] += d['rating']
#     globalGenreAvg[gid] += d['rating']
#     totalReviews[gid] += 1
# # compute user averages
# for u in genresPerUser.keys():
#     for i in range(0,5):
#         if genresPerUser[u][i] == 0:
#             avgGenresRatingPerUser[u][i] = 0
#         else:
#             avgGenresRatingPerUser[u][i] /= genresPerUser[u][i]
# # compute global averages            
# for i in range(0,5):
#     if totalReviews[i] == 0:
#         globalGenreAvg[i] = 0
#     else:
#         globalGenreAvg[i] /= totalReviews[i]

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stop_words = set(stopwords.words('english'))

for d in data:
  r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
  for w in r.split():
    if w not in stop_words:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

def feature1(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    for w in r.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    u = datum['user_id']
    if u in genresPerUser:
        feat += (genresPerUser[u])
#         feat += avgGenresRatingPerUser[u]
    else:
        feat += [0,0,0,0,0]
        # replace with global average rating for each genre instead of 0
#         feat += globalGenreAvg
    feat.append(1) #offset
    return feat

NW = 15000 # dictionary size
words = [x[1] for x in counts[:NW]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

X = [feature1(d) for d in data]
y = [d['genreID'] for d in data]

Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]

mod = linear_model.LogisticRegression(C=1)
mod.fit(Xtrain, ytrain)

pred = mod.predict(Xvalid)
# run on test set
data_test = []

for d in readGz("test_Category.json.gz"):
    data_test.append(d)
Xtest = [feature1(d) for d in data_test]
pred_test = mod.predict(Xtest)
predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
    pos += 1

predictions.close()