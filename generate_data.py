from collections import Counter 
import json
import math
import random
import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default='./data/broadcast_news.json')
    parser.add_argument("--download_data", type=bool, default=True)
    args = parser.parse_args()
    FILE_PATH = args.data_file

    
    if args.download_data is True:
        gdd.download_file_from_google_drive(file_id='1g8NP8cLV_v6ci3S9MoXHuGxtp7N2AnUY',
                                        dest_path=args.data_file)

    random.seed(42)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open('./data/raw_text.txt', 'w', encoding='utf-8') as f:
        text = []
        for ind in data.keys():
            sample = data[ind]
            for key in sample['broadcast'].keys():
                text.append(sample['broadcast'][key])
            for key in sample['news'].keys():
                text.append(sample['news'][key]['title'])
                text.append(sample['news'][key]['body'])
        f.write('\n'.join(text))



    cleaned_data = {}
    for match_id in data.keys():
        if len(data[match_id]['news']) != 0:
            cleaned_data[match_id] = data[match_id]
    
    data = cleaned_data

    inds = list(data.keys())
    random.shuffle(inds)

    train_len = math.ceil(0.95 * len(data))

    print(len(data), train_len, len(data) - train_len)

    train_ind = inds[:train_len]
    test_ind = inds[train_len:]

    with open('./data/train.json', 'w', encoding='utf-8') as f:
        train = {}
        for ind in train_ind:
            train[ind] = data[ind]
        json.dump(train, f, ensure_ascii=False, indent=4, sort_keys=True)
    
    with open('./data/test.json', 'w', encoding='utf-8') as f:
        test = {}
        for ind in test_ind:
            test[ind] = data[ind]
        json.dump(test, f, ensure_ascii=False, indent=4, sort_keys=True)

    keys = list(train.keys())[:5]
    with open('./data/data_sample.json', 'w', encoding='utf-8') as f:
        d = {}
        for ind in keys:
            d[ind] = train[ind]
        json.dump(d, f, ensure_ascii=False, indent=4, sort_keys=True)
