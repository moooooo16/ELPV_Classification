import os
import numpy as np


def load_data():

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    data_path = os.path.join(root_path, 'data')
    # img_dir = os.path.join(data_path, 'images')
    data = np.genfromtxt(os.path.join(data_path, 'labels.csv'),
                         dtype=['|S19', '<f8', '|S4'], 
                         names=['path', 'probability', 'type'])
    
    return root_path, data_path, np.char.decode(data['path']), data['probability'], np.char.decode(data['type'])



                                             
    