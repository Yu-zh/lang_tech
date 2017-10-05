"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import ml
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
import pickle


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def predict():
    pass


if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    # train_file = 'train.txt'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []
        while queue:
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)
        stack, graph = transition.empty_stack(stack, graph)
        print('Equal graphs:', transition.equal_graphs(sentence, graph))

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        print(transitions)
        print(graph)
    # features = ['stack0_POS', 'stack0_word', 'queue0_POS', 'queue0_word', 'can-re', 'can-la']
    #
    # features = ['stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word',
    #             'queue0_POS', 'queue1_POS', 'queue0_word', 'queue1_word',
    #             'can-re', 'can-la']

    # features = ['stack0_pw_POS', 'stack0_fw_POS', 'stack0_pw_word', 'stack0_fw_word',
    # 'stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word',
    # 'queue0_POS', 'queue1_POS', 'queue0_word', 'queue1_word',
    # 'can-re', 'can-la']
    #
    # X,y = ml.extract_features(features, formatted_corpus)
    # for i in range(len(X)):
    #     print(X[i],y[i])
    #
    # y, dict_classes, inv_dict_classes = ml.encode_classes(y)
    # vec = DictVectorizer(sparse=True)
    # X = vec.fit_transform(X)
    #
    #
    # filename = "model3.sav"
    # classifier = pickle.load(open(filename, 'rb'))
    #
    #
    # # classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    # # model = classifier.fit(X, y)
    # # print(model)
    # y_predicted = classifier.predict(X)
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(y, y_predicted)))
    #
    #
    # # filename = "model3.sav"
    # # pickle.dump(model, open(filename, 'wb'))