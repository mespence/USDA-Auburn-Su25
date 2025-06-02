from statistics import mode
from collections import defaultdict

def sliding_mode(labels, window_size, pass_labels = []):
    """
    sliding_mode takes in a list of labels and at each index 
      it looks at the window around that index to create a new list
      in which the label at each index is the mode of the labels
      in the window around that index in the original list. Parts of the
      window outside of the list are ignored. In the event of a tie
      behaviour follows that of the statistics.mode function.
    3:15 - 
    Inputs:
        labels: the input list of labels
        window_size: the total width of the window centered around
          each index, must be positive
        pass_labels: labels to pass through directly to output,
          they will not be replaced by the mode around their window
    Output: a list of the same size as the input labels
    """
    output = []
    for i in range(len(labels)):
        if labels[i] in pass_labels:
            output.append(labels[i])
        else:
            half_window = (window_size - 1) // 2
            window_left = max(0, i - half_window)
            window_right = min(len(labels) - 1, i + half_window)
            # Even-sized windows are not symmetric about index
            if half_window % 2 == 0:
                window_right += 1
            output.append(mode(labels[window_left:window_right]))
    return output

def generate_adjacency_list(labels_list):
    """
    generate_adjacency_list iterates through each list in the
      input list of lists and adds every transition it sees
      to an adjacency list (which is a dictionary internally).
    Inputs:
        labels_list: a list of label lists
    Output: an adjacency list implented as a dictionary
    """
    output = defaultdict(set)
    for labels in labels_list:
        for i in range(len(labels) - 1):
            output[labels[i]].add(labels[i + 1])
    return output

def find_impossible_sequences(labels, adjacency_list):
    """
    find_impossible_sequences iterates through a list of input
      labels and returns a list of indices at which there is
      an impossible transition based on an adjacency list.
    Inputs:
        labels: the input list of labels
        adjacency_list: a dictionary of label to set of
          valid transitions from that label pairs
    Output: a list of indices at which an impossible transition
        occurs
    """
    output = []
    for i in range(len(labels) - 1):
        if labels[i + 1] not in adjacency_list[labels[i]]:
            output.append(i)
    return output

def tests():
    # sliding_mode
    assert sliding_mode([1, 2, 3], 1) == [1, 2, 3]
    assert sliding_mode([1, 2, 3], 2) == [1, 2, 3]
    assert sliding_mode([1, 2, 3], 42) == [1, 1, 1]
    test_list = [1] * 4 + [2] + [1] * 2
    assert sliding_mode(test_list, 5) == [1] * 7
    assert sliding_mode(test_list, 5, [2]) == test_list
    
    # adjacency lists and impossible sequence finding
    assert generate_adjacency_list([[1, 1]]) == {1: {1}}
    assert generate_adjacency_list([[1, 1], [1, 2]]) == {1: {1, 2}}
    test_al = generate_adjacency_list([[1, 1]])
    assert find_impossible_sequences([1] * 20 + [2], test_al) == [19]

# tests()
