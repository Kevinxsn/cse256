# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings
from DANmodels import SentimentDatasetDAN, DAN
from torch.utils.data import DataLoader


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        #X = X.float()
        if X.dtype in (torch.int64, torch.int32, torch.long):
            X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        #X = X.float()
        if X.dtype in (torch.int64, torch.int32, torch.long):
            X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--bpe_vocab', type=int, default=3000)
    parser.add_argument('--bpe_emb_dim', type=int, default=300)
    parser.add_argument('--bpe_max_len', type=int, default=10)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
            

        # choose one:
        emb_path = "data/glove.6B.50d-relativized.txt"
        #emb_path = "data/glove.6B.300d-relativized.txt"

        embs = read_word_embeddings(emb_path)

        train_ds = SentimentDatasetDAN("data/train.txt", embs, max_len=60)
        dev_ds   = SentimentDatasetDAN("data/dev.txt",   embs, max_len=60)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        dev_loader   = DataLoader(dev_ds,   batch_size=64, shuffle=False)

        model = DAN(
            word_embeddings=embs,
            hidden_size=200,
            num_layers=2,
            dropout=0.3,
            frozen_embeddings=False,   # try True vs False
        )

        # If your experiment() is hardcoded to lr=1e-4 and 100 epochs,
        # consider bumping lr for DAN:
        #   optimizer = Adam(..., lr=1e-3)""
        # Otherwise just call your existing experiment().
        experiment(model, train_loader, dev_loader)
    
    
    elif args.model == "BPE":
        from bpe import BytePairEncoder
        from utils import Indexer
        from DANmodels import SentimentDatasetBPE, DANSubword

        # 1) train BPE on train set words only
        train_examples = read_sentiment_examples("data/train.txt")
        all_train_words = [w for ex in train_examples for w in ex.words]

        
        print('bpe training begin')
        bpe = BytePairEncoder()
        bpe.train(all_train_words, vocab_size=args.bpe_vocab)
        print('bpe training finished')

        # 2) build subword vocabulary (Indexer) from train encodings only
        subword_indexer = Indexer()
        subword_indexer.add_and_get_index("PAD")  # 0
        subword_indexer.add_and_get_index("UNK")  # 1

        for ex in train_examples:
            sws = bpe.encode_sentence(ex.words)
            for sw in sws:
                subword_indexer.add_and_get_index(sw)

        # 3) datasets/loaders
        train_ds = SentimentDatasetBPE("data/train.txt", bpe, subword_indexer, max_len=args.bpe_max_len)
        dev_ds   = SentimentDatasetBPE("data/dev.txt",   bpe, subword_indexer, max_len=args.bpe_max_len)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        dev_loader   = DataLoader(dev_ds,   batch_size=64, shuffle=False)

        # 4) model (random embeddings, per instructions)
        model = DANSubword(
            vocab_size=len(subword_indexer),
            emb_dim=args.bpe_emb_dim,
            hidden_size=200,
            num_layers=2,
            dropout=0.3,
            pad_idx=0
        )

        experiment(model, train_loader, dev_loader)

if __name__ == "__main__":
    main()
