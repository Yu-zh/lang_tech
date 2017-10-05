import pickle
import conll
import ml
import dparser
import transition
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer


def extract1(stack, queue, graph):
    if len(stack) == 0:
        Si = ['nil', 'nil']
    else:
        Si = [stack[0]['postag'], stack[0]['form']]
    Qi = [queue[0]['postag'], queue[0]['form']]
    return Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]

def extract2(stack, queue, graph):
    if len(stack) == 0:
        Si = ['nil', 'nil', 'nil', 'nil']
    elif len(stack) == 1:
        Si = [stack[0]['postag'], 'nil', stack[0]['form'], 'nil']
    else:
        Si = [stack[0]['postag'], stack[1]['postag'], stack[0]['form'], stack[1]['form']]
    if len(queue) == 1:
        Qi = [queue[0]['postag'], 'nil', queue[0]['form'], 'nil']
    else:
        Qi = [queue[0]['postag'], queue[1]['postag'], queue[0]['form'], queue[1]['form']]
    return Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]

def extract3(stack, queue, graph, sentence):
    if len(stack) == 0 or len(sentence) == 1:
        Pi = ['nil', 'nil', 'nil', 'nil']
    else:
        id = int(stack[0]['id'])
        if id == 0:
            Pi = ['nil', 'nil', 'nil', 'nil']
        elif id == 1:
            Pi = ['nil', sentence[id + 1]['postag'], 'nil', sentence[id + 1]['form']]
        elif id == len(sentence):
            Pi = [sentence[id - 1]['postag'], 'nil', sentence[id - 1]['form'], 'nil']
        else:
            Pi = [sentence[id - 1]['postag'], sentence[id + 1]['postag'], sentence[id - 1]['form'],
                  sentence[id + 1]['form']]
    if len(stack) == 0:
        Si = ['nil', 'nil', 'nil', 'nil']
    elif len(stack) == 1:
        Si = [stack[0]['postag'], 'nil', stack[0]['form'], 'nil']
    else:
        Si = [stack[0]['postag'], stack[1]['postag'], stack[0]['form'], stack[1]['form']]
    if len(queue) == 1:
        Qi = [queue[0]['postag'], 'nil', queue[0]['form'], 'nil']
    else:
        Qi = [queue[0]['postag'], queue[1]['postag'], queue[0]['form'], queue[1]['form']]
    return Pi + Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]

def feature_extract(sentence, stack, queue, graph, feature_names):
    if len(feature_names)==6:
        return dict(zip(feature_names, extract1(stack, queue, graph)))
    elif len(feature_names)==10:
        return dict(zip(feature_names, extract2(stack, queue, graph)))
    elif len(feature_names)==14:
        return dict(zip(feature_names, extract3(stack, queue, graph, sentence)))
    else:
        return 0


def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == "ra":
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, "ra"
    elif stack and trans[:2] == "la":
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, "la"
    elif stack and trans[:2] == "re":
        return stack[1:], queue, graph, "re"
    elif trans[:2] == "sh":
        return [queue[0]] + stack, queue[1:], graph, "sh"
    else:
        print("tran error")
        return


def predict(vec, corpus, feature_names):
    f_out = open("out","w")
    clf = pickle.load(open(model_name,"rb"))
    for sentence in corpus:
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []
        while queue:
            X = feature_extract(sentence, stack, queue, graph, feature_names)
            X = vec.transform(X)
            y = clf.predict(X)
            trans = dict_classes[y[0]]
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)
        stack, graph = transition.empty_stack(stack, graph)
        for word in sentence[1:]:
            for v in word.values():
                f_out.write(v+"\t")
            id = word["id"]
            try:
                f_out.write(graph['heads'][id]+'\t'+graph['deprels'][id]+'\t_\t_\n')
            except:
                print("key error")
                f_out.write("\n")
        f_out.write("\n")


if __name__ == "__main__":
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sentences_test = conll.read_sentences(test_file)
    formatted_corpus_test = conll.split_rows(sentences_test, column_names_2006_test)

    # features = ['stack0_POS', 'stack0_word', 'queue0_POS', 'queue0_word', 'can-re', 'can-la']

    # features = ['stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word',
    #             'queue0_POS', 'queue1_POS', 'queue0_word', 'queue1_word',
    #             'can-re', 'can-la']

    features = ['stack0_pw_POS', 'stack0_fw_POS', 'stack0_pw_word', 'stack0_fw_word',
    'stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word',
    'queue0_POS', 'queue1_POS', 'queue0_word', 'queue1_word',
    'can-re', 'can-la']

    X, y = ml.extract_features(features, formatted_corpus)
    y, dict_classes, inv_dict_classes = ml.encode_classes(y)
    vec = DictVectorizer(sparse=True)
    vec.fit_transform(X)

    if len(features)==6:
        model_name = "model1.sav"
    elif len(features)==10:
        model_name = "model2.sav"
    elif len(features)==14:
        model_name = "model3.sav"
    else:
        print("error in feature name")
    predict(vec, formatted_corpus_test, features)
