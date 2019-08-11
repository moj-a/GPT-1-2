from nltk.translate.bleu_score import sentence_bleu
import torch

# candidate,_ = torch.max(outputs, 1)


def BLEU_metric(reference, candidate):
    """
    BLEU_metric from nltk https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    :param candidate: An output vector of tokens.
    :param reference: A taget vector of tokens. (based on nltk code, we can have list of refrences)
    :return: BLEU scores for 1_gram, 2_gram, 3_gram, 4_gram, and score_accumulated
    """
    
    if (len(candidate) or len(reference)) < 4:
        raise ValueError("BLEU requires a seq length of greater than 4")
    
    #convert imputs to list type
    
    if not isinstance(candidate,list):
        candidate = candidate.tolist()
    
    if not isinstance(reference,list):
        reference = reference.tolist()    
    
    
    # we need list of refrences in BLEU_nltk
    reference =[reference]
    
    score_1 = sentence_bleu(reference, candidate, weights=(1, 0 , 0, 0))
    score_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    score_3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    score_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    score_accom = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    return score_1, score_2, score_3, score_4, score_accom


def NIST_metric(candidate, reference):
    
    """
    NIST_metric from nltk  https://www.nltk.org/_modules/nltk/translate/nist_score.html
    :param candidate: An output vector of tokens. 
    :param reference: A taget vector of tokens. (based on nltk code, we can have list of refrences)
    :return: NIST score for n=5 (highest n-gram order)
    """
    
    if (len(candidate)) < 5:
        raise ValueError("NIST requires a seq length of greater than 5 for candidate")
    
    if not isinstance(candidate,list):
        candidate = candidate.tolist()
    
    if not isinstance(reference,list):
        reference = reference.tolist()    
    
    reference =[reference]
    NIST_score = sentence_nist(reference, candidate, n=5)
   
    return NIST_score



# to do list:


#ROUGE is a modification of BLEU that focuses on recall rather than precision. In other words, it looks at how many n-grams in the reference translation show up in the output, rather than the reverse.


# list of several Metrics:

#https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213