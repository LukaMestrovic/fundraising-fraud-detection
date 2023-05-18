import json

def data_preprocessing(path = "../../data/raw.txt"):

    # Open text file in read mode
    file = open(path, "r")
    
    # Read whole file to a string
    raw = file.read()
    
    # Close file
    file.close()

    raw = json.loads(raw)

    processed = {
        "mail": [ad["mail"] for ad in raw],
        "text": [ad["text"] for ad in raw],
        "label": [ad["label"] for ad in raw],
    }

    return processed

print(data_preprocessing())