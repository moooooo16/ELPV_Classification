import os
import numpy as np


def get_paths():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    data_path = os.path.join(root_path, 'data')
    out_path = os.path.join(root_path, 'out')
    
    return root_path, data_path, out_path

ROOT_PATH, DATA_PATH, OUT_PATH = get_paths()

def load_data(data_path=DATA_PATH):
    if data_path is None:
        raise Exception('Data path is not set')
    # img_dir = os.path.join(data_path, 'images')
    data = np.genfromtxt(os.path.join(data_path, 'labels.csv'),
                         dtype=['|S19', '<f8', '|S4'], 
                         names=['path', 'probability', 'type'])
    
    label = combine_labels(data['probability'], np.char.decode(data['type']))
    
    return  np.char.decode(data['path']), data['probability'], np.char.decode(data['type']), label



def combine_labels(prob, types):
    
    prob_matching = {
        0.0: 0,
        0.3333333333333333: 1,
        0.6666666666666666: 2,
        1.0: 3
    }
    
    types_matching = {
        "poly": 0,
        'mono': 1   
    }
    
    labels = []
    
    for p, t in zip(prob, types):
        labels.append(prob_matching[p] if types_matching[t] else 4 + prob_matching[p])
    
    count, unique = np.unique(labels, return_counts=True)
    print(count, unique)
    return labels
                                             
    