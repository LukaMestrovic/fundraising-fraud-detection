import json
import random

def data_preprocessing(path):

    # Open text file in read mode
    file = open(path, "r")
    
    # Read whole file to a string
    raw = file.read()
    
    # Close file
    file.close()

    raw = json.loads(raw)
    random.shuffle(raw)

    slice = int(0.8 * len(raw))

    processed_train = {
        "mail": [ad["mail"] for ad in raw[:slice]],
        "text": [ad["text"] for ad in raw[:slice]],
        "label": [ad["label"] for ad in raw[:slice]]
    }

    processed_val = {
        "mail": [ad["mail"] for ad in raw[slice:]],
        "text": [ad["text"] for ad in raw[slice:]],
        "label": [ad["label"] for ad in raw[slice:]]
    }

    return (processed_train, processed_val)