import itertools
import numpy as np
import string
import warnings

import time
import classifier_feature_util as clfutil

import emoji
from nltk import ngrams

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn_crfsuite import CRF

from corpus import Corpus


#### BEGIN K-FOLD CROSS-VALIDATION ####################################################################################

def k_fold_cross_validation(X, y, clf, k=10, shuffle=False):
    """Perform k-fold cross-validation on classifier clf using input samples X and targets y.

    Returns a tuple of:
      - a list of batches of test samples (list of NumPy-arrays)
      - a list of batches of targets (list of NumPy-arrays)
      - a list of batches of predictions (list of NumPy-arrays)
    sorted by rounds of cross-validation (i.e. the first element of each list is a NumPy-array containing the samples/
    /targets/predictions from the first round).
    X and y must be iterable, and clf must have methods fit and predict, such that clf.fit(X, y) and clf.predict(X)
    train the classifier or classify samples respectively.
    Pass shuffle=True to shuffle samples before splitting.
    """
    # Convert X and y to NumPy-arrays to make sure we can index them with arrays. (Note that converting to list first
    # may be necessary, e.g. to collect the elements produced by a generator instead of the generator itself.)
    X = np.array(list(X), dtype=object)
    y = np.array(list(y), dtype=object)

    sample_list, target_list, pred_list = [], [], []

    kf = KFold(n_splits=k, shuffle=shuffle)
    for train_idxs, test_idxs in kf.split(X):
        X_train = X[train_idxs]
        y_train = y[train_idxs]
        X_test = X[test_idxs]
        y_test = y[test_idxs]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        sample_list.append(X_test.copy())
        target_list.append(y_test.copy())
        pred_list.append(y_pred.copy())

    return sample_list, target_list, pred_list

#### END K-FOLD CROSS-VALIDATION ######################################################################################


#### BEGIN CRF-SPECIFIC STUFF #########################################################################################

def print_crf_metrics(label_list, sample_list, target_list, pred_list):
    # Temporarily disable all warnings to suppress warnings about ill-defined metrics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        num_of_cats = len(label_list)
        k = len(sample_list)

        tag_id_dictionary = dict()
        for index in range(len(label_list)):
            tag = label_list[index]
            tag_id_dictionary[tag] = index

        support=np.array([0]*num_of_cats)
        precision_score_arr,recall_score_arr=np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats)
        acc_score_arr_sen,precision_score_arr_sen,recall_score_arr_sen=np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats)
        f1_macro,precision_macro,recall_macro=0.0,0.0,0.0
        f1_weighted,precision_weighted,recall_weighted=0.0,0.0,0.0
        f1_micro,precision_micro,recall_micro=0.0,0.0,0.0
        acc_score=0.0
        acc_score_sen_level=0.0
        total_supp=0

        for X_test, y_test, y_pred in zip(sample_list, target_list, pred_list):
            y_test_f = []
            for sent_sublist in y_test.tolist():
                for tag_val in sent_sublist:
                    y_test_f.append(tag_val)
            y_pred_f = []
            for sent_sublist in y_pred:
                for tag_val in sent_sublist:
                    y_pred_f.append(tag_val)

            p, r, f1, s = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,labels=label_list,average=None)
            for i in range(0,num_of_cats) :
                precision_score_arr[i]+=p[i]
                recall_score_arr[i]+=r[i]
                support[i]+=s[i]

            acc_score+=metrics.accuracy_score(y_test_f,y_pred_f)
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='macro')
            precision_macro+=p
            recall_macro+=r
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='micro')
            precision_micro+=p
            recall_micro+=r
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='weighted')
            precision_weighted+=p
            recall_weighted+=r
            total_supp=sum(support)
            '''CRF when converting tags to binary (sentence level):'''
            y_pred_sen_level= []
            test_y_sents_sen_level=[]

            for tags in y_pred:
                binary_tags=[0]*num_of_cats
                for tag in tags:
                    binary_tags[tag_id_dictionary[tag]]=1
                y_pred_sen_level.append(binary_tags)
            for tags in y_test:
                binary_tags=[0]*num_of_cats
                for tag in tags:
                    binary_tags[tag_id_dictionary[tag]]=1
                test_y_sents_sen_level.append(binary_tags)

            for i in range(0, num_of_cats):
                y_true=[sen[i] for sen in test_y_sents_sen_level]
                y_pred=[sen[i] for sen in y_pred_sen_level]
                precision_score_arr_sen[i]+=metrics.precision_score(y_true, y_pred)
                recall_score_arr_sen[i]+=metrics.recall_score(y_true, y_pred)
                acc_score_arr_sen[i]+=metrics.accuracy_score(y_true, y_pred)
            acc_score_sen_level+=metrics.accuracy_score(test_y_sents_sen_level,y_pred_sen_level)
        precision_macro=precision_macro/k
        recall_macro=recall_macro/k
        f1_macro=2.0*precision_macro*recall_macro/(precision_macro+recall_macro)
        precision_weighted=precision_weighted/k
        recall_weighted=recall_weighted/k
        f1_weighted=2.0*precision_weighted*recall_weighted/(precision_weighted+recall_weighted)
        precision_micro=precision_micro/k
        recall_micro=recall_micro/k
        f1_micro=2.0*precision_micro*recall_micro/(precision_micro+recall_micro)

        #pretty print
        print('{0}-fold WORD-level:'.format(k))
        print("---- {0} fold cross validation of the model----".format(k))
        print(acc_score/k)
        print('Category     precision     recall      f1-score      support')
        for i in range(0, num_of_cats):
            precision_score=precision_score_arr[i]/k
            recall_score=recall_score_arr[i]/k
            f1_score=2*precision_score*recall_score/(recall_score+precision_score)
            print(' {0}  :  {1:.2f}         {2:.2f}      {3:.2f}        {4}  '.format(label_list[i],precision_score,recall_score,f1_score,support[i]))
        print('micro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_micro,recall_micro,f1_micro,total_supp))
        print('macro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_macro,recall_macro,f1_macro,total_supp))
        print('weighted avg:{0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_weighted,recall_weighted,f1_weighted,total_supp))


        print('{0}-fold CRF indirect  sentence level:'.format(k))
        print("total acc score = {0:.3f}".format(acc_score_sen_level/k))
        print('Category     accuracy     precision    recall      f1-score')
        for i in range(0, num_of_cats):
            precision_score=precision_score_arr_sen[i]/k
            recall_score=recall_score_arr_sen[i]/k
            f1_score=2*precision_score*recall_score/(recall_score+precision_score)
            acc_score=acc_score_arr_sen[i]/k
            print('{0} :    {1:.2f}          {2:.2f}         {3:.2f}         {4:.2f} '.format(label_list[i],acc_score,precision_score,recall_score,f1_score))


def word2features(sent, i, most_freq_ngrams=[]):
    """
    :param sent: the sentence
    :param i: the index of the token in sent
    :param tags: the tags of the given sentence (sent)
    :return: the features of the token at index i in sent
    """
    word = sent[i]

    lower_word = word.lower()
    list_of_ngrams = list(ngrams(lower_word, 2)) + list(ngrams(lower_word, 3))
    list_of_ngrams = [''.join(ngram) for ngram in list_of_ngrams]


    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word_with_digit': any(char.isdigit() for char in word) and word.isnumeric() is False,
        'word_pure_digit': word.isnumeric(),
        'word_with_umlaut': any(char in "üöäÜÖÄß" for char in word),
        'word_with_punct': any(char in string.punctuation for char in word),
        'word_pure_punct': all(char in string.punctuation for char in word),
        'frequent_en_word': lower_word in clfutil.FreqLists.EN_WORD_LIST,
        'frequent_de_word': lower_word in clfutil.FreqLists.DE_WORD_LIST,
        'frequent_ngrams_de': any(ngram in clfutil.MOST_COMMON_NGRAMS_DE for ngram in list_of_ngrams),
        'frequent_ngrams_en': any(ngram in clfutil.MOST_COMMON_NGRAMS_EN for ngram in list_of_ngrams),
        'is_in_emoticonlist': lower_word in clfutil.OtherLists.EMOTICON_LIST,
        'is_emoji': any(char in emoji.EMOJI_DATA for char in word),

        #derivation and flextion
        'D_Der_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_A_suf_dict.values()))),
        'D_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_N_suf_dict.values()))),
        'D_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.D_DER_V_pref_list),
        'E_Der_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_DER_A_suf_list),
        'E_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_N_suf_dict.values()))),
        'E_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.E_DER_V_pref_list),
        'D_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_V_suf_dict.values()))),
        'E_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_V_suf_dict.values()))),
        'D_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_FLEX_A_suf_dict.values()))),
        'D_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_N_suf_list),
        'D_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_V_suf_list),
        'E_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_A_suf_list),
        'E_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_N_suf_list),
        'E_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_V_suf_list),
        'D_Flex_V_circ': lower_word.startswith("ge") and (lower_word.endswith("en") or lower_word.endswith("t")),

        #NE:
        'D_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Demo_suff),
        'D_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Morph_suff),
        'E_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Demo_suff),
        'E_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Morph_suff),
        'O_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_Morph_suff),
        'D_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.D_NE_parts),
        'E_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.E_NE_parts),
        'O_NE_parts': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_suff),

        #entity lists
        'D_NE_REGs': any(w in lower_word for w in clfutil.NELists.D_NE_REGs)
                     or lower_word in clfutil.NELists.D_NE_REGs_abbr,
        'E_NE_REGs': any(w in lower_word for w in clfutil.NELists.E_NE_REGs)
                     or lower_word in clfutil.NELists.E_NE_REGs_abbr,
        'O_NE_REGs': any(w in lower_word for w in clfutil.NELists.O_NE_REGs)
                     or lower_word in clfutil.NELists.O_NE_REGs_abbr
                     or any(lower_word.startswith(w) for w in clfutil.NELists.O_REG_demonym_verisons),

        'D_NE_ORGs': lower_word in clfutil.NELists.D_NE_ORGs,
        'E_NE_ORGs': lower_word in clfutil.NELists.E_NE_ORGs,
        'O_NE_ORGs': lower_word in clfutil.NELists.O_NE_ORGs,

        'D_NE_VIPs': lower_word in clfutil.NELists.D_NE_VIPs,
        'E_NE_VIPs': lower_word in clfutil.NELists.E_NE_VIPs,
        'O_NE_VIPs': lower_word in clfutil.NELists.O_NE_VIPs,

        'D_NE_PRESS': lower_word in clfutil.NELists.D_NE_PRESS,
        'E_NE_PRESS': lower_word in clfutil.NELists.E_NE_PRESS,
        'O_NE_PRESS': lower_word in clfutil.NELists.O_NE_PRESS,

        'D_NE_COMPs': lower_word in clfutil.NELists.D_NE_COMPs,
        'E_NE_COMPs': lower_word in clfutil.NELists.E_NE_COMPs,
        'O_NE_COMPs': lower_word in clfutil.NELists.O_NE_COMPs,

        'NE_MEASURE': any(w in lower_word for w in clfutil.NELists.NE_MEASURE),

        'D_CULT': any(w in lower_word for w in clfutil.CultureTerms.D_CULT),
        'E_CULT': any(w in lower_word for w in clfutil.CultureTerms.E_CULT),
        'O_CULT': any(w in lower_word for w in clfutil.CultureTerms.O_CULT),

        'D_FuncWords': lower_word in clfutil.FunctionWords.deu_function_words,
        'E_FuncWords': lower_word in clfutil.FunctionWords.eng_function_words,

        'Interj_Word': lower_word in clfutil.OtherLists.Interj_Words,

        'URL': any(lower_word.startswith(affix) for affix in clfutil.OtherLists.URL_PREF) or any(lower_word.endswith(affix) for affix in clfutil.OtherLists.URL_SUFF) or any(affix in lower_word for affix in clfutil.OtherLists.URL_INFIX)
    }

    for ngram in most_freq_ngrams:
        features[ngram] = ngram in list_of_ngrams

    if i > 0:
        pass
    else:
        features['BOS'] = True

    if i == len(sent) - 1:
        features['EOS'] = True

    return features


def sent2features(sent, most_freq_ngrams=[]):
    """
    This function returns a list of features of each token in the given sentence (and using the corresponding tags)
    """
    return [word2features(sent, i, most_freq_ngrams) for i in range(len(sent))]

def get_ngrams(word_list, num_of_ngrams):
    ngrams_dict = dict()
    for word in word_list:
        ngram_list = [''.join(ngram) for ngram in list(ngrams(word, 2)) + list(ngrams(word, 3))]
        for ngram in ngram_list:
            if ngram in ngrams_dict.keys():
                ngrams_dict[ngram] += 1
            else:
                ngrams_dict[ngram] = 1
    sorted_list = sorted(ngrams_dict.items(), key=lambda item: item[1],reverse=True)

    res_lst = [strng for strng, value in sorted_list[:num_of_ngrams]]
    return res_lst

#### END CRF-SPECIFIC STUFF ###########################################################################################


#### BEGIN MAIN CODE ##################################################################################################

LABEL_LIST_FULL = ['1', '2',
                   '3', '3a', '3a-E', '3a-D', '3a-AE', '3a-AD', '3b', '3c', '3c-C', '3c-M', '3c-EC', '3c-EM',
                   '3-D', '3-E', '3-O',
                   '4', '4a', '4b', '4b-E', '4b-D', '4c', '4d', '4d-E', '4d-D', '4e-E',
                   '<punct>', '<EOS>', '<EOP>', '<url>']
LABEL_LIST_COLLAPSED = ['E', 'D', 'M', 'SE', 'SD', 'SO', 'O']

def main_crf_cross_validation(file_name, label_list, num_rounds):
    corpus = Corpus(file_name)

    # Find most frequent N-grams.
    word_list, _ = corpus.get_tokens()
    most_freq_ngrams = get_ngrams(word_list, 200)

    # Preprocess data: toks is a list containing one list of tokens for each sentence in the corpus, tags is the
    # corresponding list of lists of tags. We extract features from toks using sent2features(), the tags are already
    # the targets we want.
    toks, tags = corpus.get_sentences()
    X = [sent2features(s, most_freq_ngrams) for s in toks]
    y = tags

    # Initialize classifier and perform 10-fold cross-validation.
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    print_crf_metrics(label_list, *k_fold_cross_validation(X, y, crf, k=num_rounds, shuffle=True))

def main_crf_tag_new_data(tag_file, train_file):
    tag_corpus = Corpus(tag_file)
    train_corpus = Corpus(train_file)

    # Find most frequent N-grams in training data.
    word_list, _ = train_corpus.get_tokens()
    most_freq_ngrams = get_ngrams(word_list, 200)

    # Preprocess training data. See main_crf_cross_validation() for more info on what is going on here.
    train_toks, train_tags = train_corpus.get_sentences()
    X = [sent2features(s, most_freq_ngrams) for s in train_toks]
    y = train_tags

    # Train classifier.
    print("start classfier fit")
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X, y)

    start_time = time.time()

    # Predict tags for new data. We extract indices along with the tokens so we can update the tags later.
    print("start predict")
    idxs, new_toks, _ = tag_corpus.get_sentences(index=True)
    X_new = [sent2features(s, most_freq_ngrams) for s in new_toks]
    y_new = crf.predict(X_new)

    # Update tags in corpus, running one bulk update per sentence.
    for i, new_tags in zip(idxs, y_new):
        tag_corpus.set_tags(i, new_tags)

    # Write corpus with new tags back to CSV file.
    tag_corpus.to_csv(tag_file)

    end_time = time.time()
    print("Time:" + str(int(end_time - start_time)))

#### END MAIN CODE ####################################################################################################







if __name__ == "__main__":
    main_crf_cross_validation("corpus_manu.csv", LABEL_LIST_FULL, 10) # 10-fold cross-validation
    main_crf_cross_validation("corpus_manu_collapsed.csv", LABEL_LIST_COLLAPSED, 10) # 10-fold cross-validation
    main_crf_tag_new_data("corpus_auto.csv", "corpus_manu.csv")
