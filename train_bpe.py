import youtokentome as yttm 
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", default=30000)
    parser.add_argument("--raw_text_file", default='./data/raw_text.txt')
    args = parser.parse_args()

    vocab_size = args.vocab_size
    train_data_path = args.raw_text_file

    model_path = "./data/bpe_{}_size.model".format(vocab_size)

    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_path)
