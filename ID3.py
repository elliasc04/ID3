from node import Node
import math
import collections
import parse

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  def argMax_informationGain(d):

    # get frequency for each class for the convenience of later calculation for parent entropy
    frequencyDict = dict(collections.Counter(classList))
    # Calculate parent entropy
    parentEntropy = 0
    for myKey in frequencyDict:
      keyProbability = frequencyDict[myKey] / len(classList)
      parentEntropy -= keyProbability * math.log2(keyProbability)

    # Calculate information gain for each attribute
    informationGainDict = {} # hold information gain for each attribute: {att1: IG, att2: IG,...}
    for myKey in d[0]:
      if myKey == 'Class':
        continue
      childrenClass = {} # Hold split of class per value of attribute: {value1:{class1: #, class2: #,...},...}
      # Split class based on value of attribute first
      for myRow in d:
        if myRow[myKey] not in childrenClass:
          # value of attribute not in childrenClass
          childrenClass[myRow[myKey]] = {myRow["Class"]: 1}

        elif myRow["Class"] not in childrenClass[myRow[myKey]]:
          # class not in value of attribute
            childrenClass[myRow[myKey]][myRow["Class"]] = 1

        else:
          childrenClass[myRow[myKey]][myRow["Class"]] += 1

      # Calculate average entropy of children
      averageEntropyOfChildren = 0
      for att_value in childrenClass:
        totalNumOfValue = sum([childrenClass[att_value][x] for x in childrenClass[att_value]])
        # Calculate child entropy for each value of attribute
        childEntropy = 0
        for subKey in childrenClass[att_value]:
          subKeyProbability = childrenClass[att_value][subKey] / totalNumOfValue
          childEntropy -= subKeyProbability * math.log2(subKeyProbability)


        averageEntropyOfChildren += (totalNumOfValue / len(classList)) * childEntropy

      informationGain = parentEntropy - averageEntropyOfChildren
      informationGainDict[myKey] = informationGain

    return max(informationGainDict, key = informationGainDict.get)

  def mostCommonValue(d, att):
      values = [x[att] for x in d]
      return max(set(values), key = values.count)


  # put all classes in a list for later use
  classList = [x["Class"] for x in examples]

  t = Node()
  # Assign label the most common class or default passed in if classList empty
  # It is important here that examples cannot be empty unless t is a root
  t.label = max(set(classList), key=classList.count) if classList else default # default only if root with empty data
  t.weight = len(examples)

  # Use default to track attribute values and weight: {att1: {value1, value2,...}, att2,...}
  # default not a dictionary only if t is a root
  if type(default) != dict: # root
    default = {}
    for att in examples[0]:
      default[att] = set([x[att] for x in examples])

    # change unknown "?" to most common value for that attribute
    for row in examples:
      for key in row:
        if row[key] == "?":
            row[key] = mostCommonValue(examples, key)

  # All labels are the same class or attribute empty
  if len(set(classList)) == 1 or len(examples[0]) == 1:
    return t

  else:
    # Get the attribute that best classifies examples
    AStar = argMax_informationGain(examples)

    # Create subtree for each possible value "a" for A*
    for a in default[AStar]:
      # Get subset where A* == a and remove A* from examples
      subsetOfExample = [{key: val for key, val in x.items() if key != AStar} for x in examples if x[AStar] == a]

      if subsetOfExample:
        t.label = AStar
        t.children[a] = ID3(subsetOfExample, default)
      else: # D_a is empty, then add a leaf node with label of the most common value
          # This occurs when there are still features left but no data because
          # data is removed when subtracting data for other features
          leaf = Node()
          leaf.label = t.label
          leaf.weight = 0
          t.children[a] = leaf

  return t


def prune(node, examples):
    """
    Takes in a decision tree and a dataset for validation. Performs reduced error pruning
    to simplify the tree based on validation accuracy.
    """

    def prune_subtree(node, validation):
        if not node.children:
          return node

        #organize validation set into dict, grouping by value of node.label
        subdictset = {}
        for item in validation:
          if item[node.label] not in subdictset.keys():
            subdictset[item[node.label]] = [item]
          else:
            subdictset[item[node.label]].append(item)

        #if node child shares a key with the subdictset, recursively prune that child using the matching subdictset value as validation.
        #else, recursively prune it with no validation data
        for child in node.children.keys():
          if child in subdictset.keys():
              prune_subtree(node.children[child], subdictset[child])
          else:
              prune_subtree(node.children[child], {})

        #calculate the mode in the 'Class' attribute of the validation set. if validation is empty, prune the leaf.
        mode = None
        if len(validation) > 0:
          class_values = [item['Class'] for item in validation]
          counter = collections.Counter(class_values)
          mode = counter.most_common(1)[0][0]
        elif len(validation) == 0:
          node.children = {}
          node.label = mode
          return node

        #calculate the local accuracy using Counter.
        counts = collections.Counter(item['Class'] for item in validation)
        localct = counts[mode]

        #if the local accuracy is higher than the overall tree accuracy, prune the leaf.
        if localct / len(validation) >= test(node, examples):
          node.children = {}
          node.label = mode
    prune_subtree(node, examples)  # Call the pruning function on the root node
    # print_tree(node)







def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples. Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    failed = 0
    for x in examples:
        target = evaluate(node, x)
        if target != x['Class']:
            failed = failed + 1
    return 1 - failed/len(examples)  #accuracy


def evaluate(node, example):
    '''
    Takes in a tree and one example. Returns the Class value that the tree
    assigns to the example.
    '''
    if not node.children:
        return node.label  # Return the Class value of the leaf node
    else:
        xvalue = example[node.label]
        if node.children.get(xvalue) is not None:
            return evaluate(node.children[xvalue], example)
        else:
            return node.label

def print_tree(node, indent=""):
    """
    Recursively print the decision tree structure.
    """
    if not node.children:
        print(indent + "Class: " + str(node.label))
    else:
        print(indent + node.label)
        for value, child_tree in node.children.items():
            print(indent + "  └─ " + str(value))
            print_tree(child_tree, indent + "    ")
