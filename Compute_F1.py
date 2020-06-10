import numpy as np
import scipy.stats
from collections import Counter
import itertools
import scipy.stats as stats

scores = [84.37, 83.93, 84.61, 84.73, 84.68] ## spanish best dev scores from 5 runs
# print (np.mean(scores))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h


def compute_accuracy(gold_labels, pred_labels):
    accuracy = 0
    correct_sent = []
    # print (gold_labels[0:10], pred_labels[0:10])
    for ind, val in enumerate(gold_labels):
        if gold_labels[ind] == pred_labels[ind]:
           accuracy+=1
           correct_sent.append(ind)
    return accuracy/float(len(gold_labels)), correct_sent

def get_differences_list(list1):
    list1 = [int(i1) for i1 in list1]
    sorted_list1 = sorted(list1)[::-1] ## descending order
    difference_list = []
    for ind1 in range(len(sorted_list1)-1):
        difference_list.append(sorted_list1[ind1]-sorted_list1[ind1+1])

    return difference_list

def get_negative_flag(list1): ## this flag will be used to determine if the coherence sequence is opposite to the one in the given passage
    list1 = [int(i1) for i1 in list1]

    negative_flag = 0
    for i,val in enumerate(list1[:-1]):
        if val - list1[i+1] > 0:
           negative_flag = 1
           break

    return negative_flag


def F1_Score_individual_just(gold_labels, pred_labels_orig):

    ## first converting indexes of predicted labels to integer from string:
    pred_labels = [int(s1) for s1 in pred_labels_orig]

    precision = len(set(gold_labels).intersection(set(pred_labels)))/float(max(1,len(pred_labels)))
    recall = len(set(gold_labels).intersection(set(pred_labels)))/float(len(gold_labels))
    if precision == 0 and recall == 0:
        F1_score = 0
    else:
        F1_score = (2 * precision * recall) / float(precision + recall)

    return  [precision, recall,F1_score]



def F1_Score_Rouge(gold_labels, pred_labels):

    precision = len(set(gold_labels).intersection(set(pred_labels)))/float(max(1,len(pred_labels)))
    recall = len(set(gold_labels).intersection(set(pred_labels)))/float(len(gold_labels))

    if precision == 0 and recall == 0:
        F1_score = 0
    else:
        F1_score = (2 * precision * recall) / float(precision + recall)

    return  [precision, recall, F1_score]


def F1_Score_just_quality(gold_labels, pred_labels):
    precision = 0
    recall = 0
    Exact_match = 0

    gold_label_gaps = []

    perfect_scored_paragraphs = []

    if len(gold_labels) == len(pred_labels):
       for gind, glabel in enumerate(gold_labels):
           gold_label_gaps += get_differences_list(glabel)

           precision += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind])))
           recall += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind]))

           if len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind]))) == 1 and len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind])) == 1:
              perfect_scored_paragraphs.append(gind)


       final_recall = recall/float(len(gold_labels))
       final_precision = precision/float(len(gold_labels))

    else:
       print ("The case should not happen, RECHECK")

    # print ("the final precision and recall are as following: ", final_precision, final_recall)

    # print("The gold lab gaps for coherence part looks like ", Counter(gold_label_gaps))
    if final_recall == 0 and final_precision == 0:
        F1_score = 0

    else:
        F1_score = (2 * final_precision * final_recall) / float( final_precision + final_recall)

    return (final_precision, final_recall, F1_score), perfect_scored_paragraphs



def kendall_tau_distance(order_a, order_b):
    pairs = itertools.combinations(range(1, len(order_a)+1), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance


def generate_ngrams(input, n):

    output = []
    for i in range(len(input)-n+1):
        output.append([str(element1) for element1 in input[i:i+n]])
    output = ["_".join(item1) for item1 in output]
    return output

def Rouge_n_scores(list1, list2): ## the n is the size of the rouge score i.e. for rouge 1, n=1, for rouge =2, n=2 and so on..
    """

    :param list1:  is the gold list of the correct sequence order of the justifications
    :param list2:  is the predicted coherent list
    :return: P,R,F1 rouge scores for different n values

    This is for an individual data point where list1 is the gold and list2 is the predicted

    """
    different_possible_sizes = list(range(1,min(len(list1), len(list2)) + 1))
    PRF_different_n_vals = {}
    Final_weighted_F1_score = 0
    for n in different_possible_sizes:
        gold_n_sized_pairs = generate_ngrams(list1, n)
        predicted_n_sized_pairs = generate_ngrams(list2,n)
        # print(F1_Score_Rouge(gold_n_sized_pairs, predicted_n_sized_pairs))

        PRF_different_n_vals.update({n:F1_Score_Rouge(gold_n_sized_pairs, predicted_n_sized_pairs)})
        if n==1:
           Final_weighted_F1_score += (1*PRF_different_n_vals[n][2])  ## increase the weights for F1 score of justification selection
        else:
           Final_weighted_F1_score += (1 * PRF_different_n_vals[n][2])
    return PRF_different_n_vals, Final_weighted_F1_score

def Rouge_n_scores_dataset(list_list1, list_list2): ## for the entire dev dataset
    """

    :param list_list1: list of gold annotation list
    :param list_list2: list of predicted val list
    :return: Final P,R and F1 coherence score for the entire dev dataset





    Final_PRF_different_n_vals = {}

    for lind1, list1 in enumerate(list_list1):
        list2 = list_list2[lind1]
        different_possible_sizes = list(range(1,min(len(list1), len(list2)) + 1))
        PRF_different_n_vals = {}
        Final_weighted_F1_score = 0
        for n in different_possible_sizes:
            gold_n_sized_pairs = generate_ngrams(list1, n)
            predicted_n_sized_pairs = generate_ngrams(list2,n)
            # print(F1_Score_Rouge(gold_n_sized_pairs, predicted_n_sized_pairs))

            PRF_different_n_vals.update({n:F1_Score_Rouge(gold_n_sized_pairs, predicted_n_sized_pairs)})
            if n==1:
               Final_weighted_F1_score += (5*n*PRF_different_n_vals[n][2])  ## increase the weights for F1 score of justification selection
            else:
               Final_weighted_F1_score += (n * PRF_different_n_vals[n][2])

    """

    Accuracy = [] ## first starting with the exact match which is the most strict eval metric. This is also same as Rouge_n where n is the max size of the list.

    Accuracy_type_1 = []
    Accuracy_type_2 = []

    for lind1, list1 in enumerate(list_list1):
        list2 = list_list2[lind1]

        negative_coh_flag = get_negative_flag(list1) ## list1 is the gold one

        if list1 == list2:
           Accuracy.append(1)
           if negative_coh_flag==1:
              Accuracy_type_2.append(1)
           else:
              Accuracy_type_1.append(1)

        else:
           Accuracy.append(0)

           if negative_coh_flag == 1:
               Accuracy_type_2.append(0)
           else:
               Accuracy_type_1.append(0)

    return sum(Accuracy)/float(len(Accuracy)), sum(Accuracy_type_1)/float(len(Accuracy_type_1)), sum(Accuracy_type_2)/float(len(Accuracy_type_2))



