import argparse
import json
import math
import os
import random

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def write_to_file(filename, data, inds):
    with open(filename, "w", encoding="utf-8") as f:
        to_write = {}
        for ind in inds:
            to_write[ind] = data[ind]
        json.dump(to_write, f, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="./data/broadcast_news.json")
    parser.add_argument("--download_data", type=bool, default=True)
    parser.add_argument("--train_size", type=float, default=0.970)
    parser.add_argument("--test_size", type=float, default=0.015)
    """
    val_size = 1 - train_size - test_size
    """
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    args = parser.parse_args()
    FILE_PATH = args.data_file

    if args.download_data is True:
        if not os.path.isdir("./data/"):
            os.mkdir("./data/")
        download_file_from_google_drive("1g8NP8cLV_v6ci3S9MoXHuGxtp7N2AnUY", args.data_file)

    random.seed(42)

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("./data/raw_text.txt", "w", encoding="utf-8") as f:
        text = []
        for ind in data.keys():
            sample = data[ind]
            for key in sample["broadcast"].keys():
                text.append(sample["broadcast"][key])
            for key in sample["news"].keys():
                text.append(sample["news"][key]["title"])
                text.append(sample["news"][key]["body"])
        f.write("\n".join(text))

    cleaned_data = {}
    for match_id in data.keys():
        if len(data[match_id]["news"]) != 0:
            cleaned_data[match_id] = data[match_id]

    data = cleaned_data

    inds = list(data.keys())
    random.shuffle(inds)

    train_len = math.ceil(args.train_size * len(data))

    test_len = math.ceil(args.test_size * len(data))
    val_len = len(data) - train_len - test_len
    print(
        "Total data size: {}, train size: {}, val size: {}, test size: {}".format(
            len(data), train_len, val_len, test_len
        )
    )

    train_ind = inds[:train_len]
    assert len(train_ind) == train_len
    test_ind = inds[train_len : train_len + test_len]
    assert len(test_ind) == test_len
    val_ind = inds[train_len + test_len :]
    assert len(val_ind) == val_len

    write_to_file("./data/train.json", data, train_ind)
    write_to_file("./data/test.json", data, test_ind)
    write_to_file("./data/val.json", data, val_ind)

    keys = train_ind[:5]
    write_to_file("./data/data_sample.json", data, keys)
