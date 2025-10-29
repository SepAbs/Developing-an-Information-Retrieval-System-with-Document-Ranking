import nltk
from nltk.tokenize import wordpunct_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from rank_bm25 import BM25Okapi
from math import prod, ceil, floor
from random import uniform

def preProcessor(String):
    null, Consonants, Irregulars, stopWords, Punctuation = "", "qwrtypsdfghjklzxcvbnm", {"begun": "begin", "frozen": "freeze", "children": "child", "feet": "foot", "teeth": "tooth","mice": "mouse", "people": "person"}, stopwords.words('english'), punctuation + "/b.><"
    String = wordpunct_tokenize(String) #Case Folding & Tokenisations
    String = [Token.lower() for Token in String if Token not in Punctuation and Token.lower() not in stopWords]
    lengthString = len(String)
    
    for Index in range(lengthString):
        #Asymmetric Expansion
        if String[Index].endswith("ies") and String[Index][-4] in Consonants:
            String[Index] = String[Index][:-3] + "y"

        elif String[Index].endswith("ves"):
            String[Index] = String[Index][:-3] + "f"
            
        elif String[Index].endswith("es") and (String[Index][-3] in ["s", "x", "z"] or String[Index][-4:-2] in ["ch", "sh"] or (String[Index][-3] == "o" and String[Index][-4] in Consonants)):
            String[Index] = String[Index][:-2]
        
        elif String[Index].endswith("s"):
            String[Index] = String[Index][:-1]
            
        elif String[Index].endswith("men"):
            String[Index] = String[Index][:-3] + "man"

        if String[Index].endswith("ed"):
            String[Index] = String[Index][:-2]
            
        elif String[Index] in Irregulars.keys():
            String[Index] = Irregulars[String[Index]]

        elif String[Index].endswith("en"):
            String[Index] = String[Index][:-2]
 
    while null in String:
        String.remove(null)
        
    return String

def spaceVector(Query, Bound):
    tfidfVectorizer, q, vectorScore, retrievedDocs = TfidfVectorizer(), [" ".join(Query)], {}, []
    d = tfidfVectorizer.fit_transform(Documents)
    q = tfidfVectorizer.transform(q)
    Scores = cosine_similarity(q, d) # Calculate the cosine similarity between the query and each document and append all of the scores in a list, respectively.
    for Index in range(1400):
        vectorScore[Index] = Scores[0][Index]
        
    Ids, Scores = list(vectorScore.keys()), list(vectorScore.values())
    electedScores = nlargest(Bound, Scores) # Max-heap implemention!
    return [{"Id": Ids[Scores.index(Score)] + 1, "Score": Score} for Score in electedScores] # Retrieved documents.

def Probabilistic(Query, Bound):
    BMProb = BM25Okapi(InvertedIndex)
    Scores, electedDocuments = BMProb.get_scores(Query), BMProb.get_top_n(Query, Documents, Bound) # Also handles large queries by built-in funtions and methods.
    return [{"Id": Documents.index(Document) + 1, "Score": Scores[Documents.index(Document)]} for Document in electedDocuments] # Retrieved documents.

def Unigram(Query, Bound):
    if len(Query) < 100:
        Lambda = uniform(0.1, 1)
    else:
        Lambda = uniform(0, 0.1)
                
    Scores = [prod([Lambda * InvertedIndex[Index].count(Term) / len(InvertedIndex[Index]) + (1 - Lambda) * sum([Document.count(Term) for Document in InvertedIndex]) / TMc for Term in Query]) for Index in range(470)] #Unknown zero division error! 470 is the maximum number of documents in order to be considered in unigram language model.
    electedScores = reversed(sorted(Scores)[-Bound:])
    return [{"Id": Scores.index(Score) + 1, "Score": Score} for Score in electedScores] # Retrieved documents.

def Evaluation(Index, retrievedDocs):
    Coordinates, AP = {}, 0
    RDocs = set(Relevants[Index]) # Relevant items for the query
    numberRDocs = len(RDocs)
    for Measure in range(Bound): # Iterate on retrieved documents index by index ascendingly and then, calculate precision and recall and make coordinate '(Precision, Recall)' by the whole considered documents in each iterate.
        Measures = retrievedDocs[:Measure + 1] 
        tp = len(set(Measures).intersection(RDocs))
        Recall = tp / numberRDocs
        if Recall % 1000 // 100 >= 5:
            Recall = ceil(Recall * 10) / 10
        else:
            Recall = floor(Recall * 10) / 10
            
        if Recall not in Coordinates:
            Coordinates[Recall] = [tp / len(Measures)]
            continue
        
        Coordinates[Recall].append(tp / len(Measures))

    Recalls = Coordinates.keys()
    for Recall in Recalls:
        AP += max(Coordinates[Recall])
    return AP / 11
    
#Globals!
docFile, qFile, qrFile, Relevants, nonRelevants, SVAPs, Okapi25APs, UniAPs, TMc = open("cran.all.1400").read(), open("cran.qry").read(), open("cranqrel.txt"), {}, {}, [], [], [], 0
InvertedIndex, Queries = docFile.split(".I ")[1:], qFile.split(".I ")[1:] #First element will be '[]'!
#Preprocessing ONLY TITLE and TEXT!
InvertedIndex, Queries = [preProcessor(Document[Document.find(".T") + 2 : Document.find(".A")] + " " + Document[Document.find(".W") + 2:]) for Document in InvertedIndex], [preProcessor(Query[7:]) for Query in Queries]
#Total number of tokens in the collection.
for Document in InvertedIndex:
    TMc += len(Document)
    
Documents = [" ".join(Document) for Document in InvertedIndex]
for Line in qrFile:
    Line = list(map(int, Line.split()))
    if Line[2] >= 1:
        queryTopic = Line[0]
        if queryTopic in Relevants:
            Relevants[queryTopic].append(Line[1])
            continue
        
        Relevants[queryTopic] = [Line[1]]
        continue
    
    nonRelevants[queryTopic] = Line[1]

qrFile.close()
while True:
    try:
        Bound = int(input("How many documents do you need them be retrieved? "))
        if Bound >= 0:
            break

    except ValueError:
        print("Enter a non-negative integer!")
"""
print(f"{InvertedIndex}\n")
print(f"{Documents}\n")
print(f"{Relevants}\n")
print(f"{nonRelevants}\n")
"""

for Query in Queries:
    SV, Okapi25, Uni, realqIndex = spaceVector(Query, Bound), Probabilistic(Query, Bound), Unigram(Query, Bound), Queries.index(Query) + 1
    print(f"Space Vector Model retrieved {Bound} documents may be relevant with {realqIndex}th query: {SV}\nProbabilistic Model 'Okapi BM25' retrieved {Bound} documents may be relevant with {realqIndex}th query: {Okapi25}\nSmoothed Unigram Language Model {Bound} documents may be relevant with {realqIndex}th query: {Uni}")
    
    SVAPs.append(Evaluation(realqIndex, [Retrieved["Id"] for Retrieved in SV]))
    Okapi25APs.append(Evaluation(realqIndex, [Retrieved["Id"] for Retrieved in Okapi25]))
    UniAPs.append(Evaluation(realqIndex, [Retrieved["Id"] for Retrieved in Uni]))

print(f"11-point interpolated average precision for the whole queries {SVAPs}\n\nMAP = {sum(SVAPs) / len(SVAPs)}\n\n11-point interpolated average precision for the whole queries {Okapi25APs}\n\nMAP = {sum(Okapi25APs) / len(Okapi25APs)}\n\n11-point interpolated average precision for the whole queries {UniAPs}\n\nMAP = {sum(UniAPs) / len(UniAPs)}")
