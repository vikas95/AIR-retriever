import csv
count=0
lens=[]
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from collections import Counter
import string
import re
import argparse
import sys

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
# "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"] ## Lucene stopwords...

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
"""
with open('ARC_corpus/ARC-Challenge/ARC-Challenge-Dev.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        count+=1
        lens.append(len(row))


print (set(lens))
"""
def Query_boosting_sent(query_sentences, ans_sent, boosting_factor, stop_word_flag,):
    qwords = tokenizer.tokenize(query_sentences.lower())
    qwords = [lmtzr.lemmatize(w1) for w1 in qwords]

    cand_words = tokenizer.tokenize(ans_sent.lower())
    cand_words = [lmtzr.lemmatize(w1) for w1 in cand_words]


    if stop_word_flag == 1:
        qwords = [w for w in qwords if not w in stop_words]
        cand_words = [w for w in cand_words if not w in stop_words]

    new_sent = " ".join(qwords)
    for cw in cand_words:
        new_sent = new_sent + " " + cw +"^" +str(boosting_factor)
    new_sent += "\n"
    return new_sent

def Preprocess_QA_sentences_Quoref(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def Preprocess_QA_sentences(sentences, stop_word_flag):

    # words=tokenizer.tokenize(Preprocess_QA_sentences_Quoref(sentences))
    words=tokenizer.tokenize(sentences.lower())

    words=[lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag==1:
       words = [w for w in words if not w in stop_words]
    # new_sent=" ".join(words)
    # new_sent+="\n"
    return words

def Preprocess_KB_sentences(sentences, stop_word_flag):
    sentence_words = sentences.strip().split()
    BM25_score = float(sentence_words[0])
    sentences = " ".join(sentence_words[1:])
    words=tokenizer.tokenize(sentences)
    words=[lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag==1:
       words = [w for w in words if not w in stop_words]
    # new_sent=" ".join(words)
    # new_sent+="\n"
    return BM25_score, words


def Write_ARC_KB(filename, new_file, stop_word_flag):
    KB=open(filename,"r")
    new_KB=open(new_file,"w")
    count=0
    for line in KB:
        new_sent=Preprocess_KB_sentences(line.strip(), stop_word_flag)
        new_KB.write(new_sent)
        count+=1
        if count%10000==0:
           print(count)

def get_IDF_weights(file_name, IDF):
    Doc_len=[]
    Corpus=[]
    All_words=[]
    file1=open(file_name)
    for line in file1:    ####### each line is a doc
        line=line.lower()
        words=line.split()
        # words=tokenizer.tokenize(line)
        # words = [lmtzr.lemmatize(w1) for w1 in words]
        Document={}  ########## dictionary - having terms as key and TF as values of the key.
        Doc_len.append(len(words))
        unique_words=list(set(words))
        for w1 in unique_words:
            if w1 in IDF.keys():
               IDF[str(w1)]+=1
               # print ("yes, we come here", w1)
            #else:
               #IDF.update({str(w1):1})


        All_words += unique_words
        for term1 in unique_words:
            Document[str(term1)]=words.count(term1)

        Corpus.append(Document)
    All_words=list(set(All_words))
    return Doc_len, Corpus, All_words, IDF


# text = "In 1630s New England, English settler William and his family \u2014 wife Katherine, daughter Thomasin, son Caleb, and fraternal twins Mercy and Jonas \u2014 are banished from a Puritan Plymouth Colony over a religious dispute. They build a farm near a large, secluded forest and Katherine has a newborn child, Samuel. One day, Thomasin is playing peekaboo with Samuel when the baby abruptly disappears. It is revealed that a witch had stolen the unbaptized Samuel and that night kills him and uses his blood and fat to make a flying ointment.Katherine, devastated, spends her days crying and praying, while William insists a wolf stole the baby. Even though Katherine forbids the children going to the forest, William takes Caleb to lay a trap for food. Caleb asks if Samuel's unbaptized soul will reach Heaven. William chastises Caleb for raising the question and later reveals to Caleb that he traded Katherine's silver cup for hunting supplies. That night, Katherine questions Thomasin about the disappearance of her cup while implying Thomasin was responsible for Samuel's vanishing. After the children retire to bed, they overhear their parents discussing sending Thomasin away to serve another family.\nEarly the next morning, Thomasin finds Caleb preparing to check the trap in the forest. She forces Caleb to take her with him by threatening to awaken their parents. While walking in the woods, they spot a hare, which sends their horse Burt into a panic and their dog Fowler promptly chases. Caleb runs off after the pair, while the horse throws Thomasin off, knocking her unconscious. Caleb becomes lost in the woods and stumbles upon Fowler's disemboweled body. As he gets deeper into the woods, he comes across a hovel, where a beautiful young woman emerges and seduces him. William finds Thomasin and takes her home. Katherine angrily chastises Thomasin for taking Caleb into the woods and, to save Thomasin, William reluctantly admits that he sold Katherine's silver cup"
# preprocessed_text = Preprocess_QA_sentences_Quoref(text)
#
# print (preprocessed_text)


