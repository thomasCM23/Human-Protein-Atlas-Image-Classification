import numpy as np
import pandas as pd
import imageio
from sklearn.model_selection import train_test_split
import tensorflow as tf

LABEL_NAMES = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}

DATA_DIRECTORY = "data/"
TRAIN = "train/"
TEST= "test/"

reverse_train_labels = dict((v,k) for k,v in LABEL_NAMES.items())

def set_targets(data_set):
    for key in LABEL_NAMES.keys():
        data_set[LABEL_NAMES[key]] = 0
    return data_set


def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = LABEL_NAMES[int(num)]
        row.loc[name] = 1
    return row

def load_data_csv(file_name):
    # Only train has a CSV
    return pd.read_csv(DATA_DIRECTORY + file_name)

def data_to_name_targets(data):
    file_names = np.asarray(data['Id'].apply(lambda x: f"{x}_green.png").tolist())
    targets = data.drop(["Id","Target"], axis=1).values
    return file_names, targets

def train_val_test_split(file_names, targets, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(file_names, targets, test_size=test_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_images(file_name_data, height=512, width=512, channels=1, start=0, end=0):
    # Load images into array, batch_size should include offset
    if(end >= file_name_data.shape[0]):
        end = file_name_data.shape[0]
    image_data = np.zeros((end-start, height, width, channels))
    for i in range(0, end-start):
        if(start+i >= file_name_data.shape[0]):
            break
        path = DATA_DIRECTORY + TRAIN + file_name_data[i+start]
        image_data[i, :, :, 0] = imageio.imread(path)
    return image_data


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

