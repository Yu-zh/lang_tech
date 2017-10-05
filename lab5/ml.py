import conll
import dparser
import transition

def extract_features(feature_names, formatted_corpus):
    X_l = []
    y_l = []
    for sentence in formatted_corpus:
        if len(feature_names)==6:
            X, y = extract_features_sent1(feature_names, sentence)
        elif len(feature_names)==10:
            X, y = extract_features_sent2(feature_names, sentence)
        elif len(feature_names)==14:
            X, y = extract_features_sent3(feature_names, sentence)
        else:
            print("error feature name")
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l

def extract_features_sent1(feature_names, sentence):
    X = []
    stack = []
    queue = list(sentence)
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    transitions = []
    while len(queue):
        if len(stack)==0:
            Si = ['nil','nil']
        else:
            Si = [stack[0]['postag'],stack[0]['form']]
        Qi = [queue[0]['postag'],queue[0]['form']]
        Xi = Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]
        stack, queue, graph, trans = dparser.reference(stack, queue, graph)
        X.append(dict(zip(feature_names, Xi)))
        transitions.append(trans)
    # stack, graph = transition.empty_stack(stack, graph)
    return X, transitions

def extract_features_sent2(feature_names, sentence):
    X = []
    stack = []
    queue = list(sentence)
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    transitions = []
    while len(queue):
        if len(stack)==0:
            Si = ['nil','nil','nil','nil']
        elif len(stack)==1:
            Si = [stack[0]['postag'],'nil',stack[0]['form'],'nil']
        else:
            Si = [stack[0]['postag'], stack[1]['postag'], stack[0]['form'], stack[1]['form']]
        if len(queue)==1:
            Qi = [queue[0]['postag'],'nil',queue[0]['form'],'nil']
        else:
            Qi = [queue[0]['postag'],queue[1]['postag'],queue[0]['form'],queue[1]['form']]
        Xi = Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]
        stack, queue, graph, trans = dparser.reference(stack, queue, graph)
        X.append(dict(zip(feature_names, Xi)))
        transitions.append(trans)
    # stack, graph = transition.empty_stack(stack, graph)
    return X, transitions

def extract_features_sent3(feature_names, sentence):
    X = []
    stack = []
    queue = list(sentence)
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    transitions = []
    while queue:
        if len(stack)==0 or len(sentence)==1:
            Pi = ['nil','nil','nil','nil']
        else:
            id = int(stack[0]['id'])
            if id==0:
                Pi = ['nil','nil','nil','nil']
            elif id==1:
                Pi = ['nil', sentence[id+1]['postag'], 'nil', sentence[id+1]['form']]
            elif id==len(sentence):
                Pi = [sentence[id-1]['postag'], 'nil', sentence[id-1]['form'], 'nil']
            else:
                Pi = [sentence[id-1]['postag'], sentence[id+1]['postag'], sentence[id-1]['form'], sentence[id+1]['form']]
        if len(stack)==0:
            Si = ['nil','nil','nil','nil']
        elif len(stack)==1:
            Si = [stack[0]['postag'],'nil',stack[0]['form'],'nil']
        else:
            Si = [stack[0]['postag'], stack[1]['postag'], stack[0]['form'], stack[1]['form']]
        if len(queue)==1:
            Qi = [queue[0]['postag'],'nil',queue[0]['form'],'nil']
        else:
            Qi = [queue[0]['postag'],queue[1]['postag'],queue[0]['form'],queue[1]['form']]
        Xi = Pi + Si + Qi + [transition.can_reduce(stack, graph), transition.can_leftarc(stack, graph)]
        stack, queue, graph, trans = dparser.reference(stack, queue, graph)
        X.append(dict(zip(feature_names, Xi)))
        transitions.append(trans)
    # stack, graph = transition.empty_stack(stack, graph)
    return X, transitions

def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))
    # We assign each name a number
    dict_classes = dict(enumerate(classes))

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes
