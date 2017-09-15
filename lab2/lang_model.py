import regex as re
import sys
import numpy as np
from count import count_unigrams
from count_bigrams import count_bigrams
import functools

def tokenize(text):
    words = re.findall('\p{L}+|<s>|</s>', text)
    return words

def add_tag(text):
    text_with_tag = re.sub("(\p{Lu}.*?[\.\!\?])", r"<s> \1 </s>", text, flags=re.S)
    # text_with_tag = re.sub(r"(\b((?!\.|\?|\!).)+\b)", r"<s> \1 </s>", text, flags=re.S)
    return text_with_tag


def rm_punc_and_lower_case(text):
    text_rm_punc = re.sub(r"(?!(/)s>)[\p{P}]","",text)
    return text_rm_punc.lower()

def probability_unigrams(sentence, unigrams_freq):
    word_count = sum(unigrams_freq.values())
    tokens = tokenize(sentence.lower()) + ['</s>']
    print("Unigram model")
    print("=====================================================")
    print("wi C(wi) #words P(wi)")
    print("=====================================================")
    prob = 1
    for token in tokens:
        p = unigrams_freq[token] / word_count
        print(token," ",unigrams_freq[token]," ",word_count," ",p)
        prob *= p
    print("=====================================================")
    print("Prob. unigrams:", prob)
    entropy_rate = -np.log2(prob)/len(tokens)
    perplexity = 2**entropy_rate
    print("entropy rate: ", entropy_rate)
    print("perplexity: ", perplexity)
    print()


def probability_bigrams(sentence, unigrams_freq, bigrams_freq):
    word_count = sum(unigrams_freq.values())
    tokens = ['<s>'] + tokenize(sentence.lower()) + ['</s>']
    bitokens = [tuple(tokens[inx:inx + 2])
               for inx in range(len(tokens) - 1)]
    print("Bigram model")
    print("=====================================================")
    print("wi wi+1 Ci,i+1 C(i) P(wi+1|wi)")
    print("=====================================================")
    prob = 1
    for bitoken in bitokens:
        if bitoken in bigrams_freq.keys():
            p = bigrams_freq[bitoken] / unigrams_freq[bitoken[0]]
            print(bitoken[0]," ",bitoken[1]," ",bigrams_freq[bitoken]," ",
                  unigrams_freq[bitoken[0]]," ",p)
            prob *= p
        else:
            p = unigrams_freq[bitoken[1]]/word_count
            print(bitoken[0], " ", bitoken[1], " 0 ",
                  unigrams_freq[bitoken[0]], " 0.0 *backoff: ", p)
            prob *= p
    print("=====================================================")
    print("Prob. bigrams: ",prob)
    entropy_rate = -np.log2(prob) / len(bitokens)
    perplexity = 2 ** entropy_rate
    print("entropy rate: ", entropy_rate)
    print("perplexity: ", perplexity)
    print()




if __name__ == "__main__":
    text = open("Selma.txt","r").read()
    text_token = tokenize(rm_punc_and_lower_case(add_tag(text)))
    # no_token = tokenize(text.lower())

    # print(add_tag(text))
    unigrams_freq = count_unigrams(text_token)
    bigrams_freq = count_bigrams(text_token)
    print("possible number of bigrams:",len(unigrams_freq)**2)
    print("real number of bigrams:",len(bigrams_freq))
    sentence = "Det var en gång en katt som hette Nils"
    probability_unigrams(sentence, unigrams_freq)
    probability_bigrams(sentence, unigrams_freq, bigrams_freq)
    # sentences = ["hon hade fått större kärlek av sina föräldrar än någon annan han visste och sådan kärlek måste vändas i välsignelse",
    #              "när prästen sa detta kom alla människor att se bort mot klara gulla och de förundrade sig över vad de såg",
    #              "prästens ord tycktes redan ha gått i uppfyllelse",
    #              "där stod klara fina gulleborg ifrån skrolycka hon som var uppkallad efter själva solen vid sina föräldrars grav och lyste som en förklarad",
    #              "hon var likaså vacker som den söndagen då hon gick till kyrkan i den röda klänningen om inte vackrare"]
    # for s in sentences:
    #     print(s+"\n")
    #     probability_unigrams(s, unigrams_freq)
    #     probability_bigrams(s, unigrams_freq, bigrams_freq)

