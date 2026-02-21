import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from transformer import TransformerEncoder
from transformer import TransformerDecoder
import torch.nn.functional as F
from utilities import Utilities


from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from sparse_attention import SparseDecoder


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate_enco = 1e-4  # Learning rate for the optimizer
learning_rate = 1e-3
n_embd = 64  # Embedding dimension
n_head = 4  # Number of attention heads
n_layer = 8  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 1000 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 75 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


'''
def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity
'''

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters: 
                break
            X, Y = X.to(device), Y.to(device)
            loss, _ = decoderLMmodel(X, Y) 
            losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss).item() 
    decoderLMmodel.train()
    return perplexity


class SimpleClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        return self.network(x)




def count_params(module):
    return sum(p.numel() for p in module.parameters())


def compute_classifier_accuracy_custom(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            embeddings, _ = encoder(X)
            pooled = embeddings.mean(dim=1)
            outputs = classifier(pooled)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    return (100 * total_correct / total_samples)



def main1():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    vocab_size = tokenizer.vocab_size

    # 1. Initialize Models
    # Using hyperparameters from main.py and suggested d_ff (4 * n_embd)
    encoder = TransformerEncoder(
        vocab_size=vocab_size, 
        d_model=n_embd, 
        n_heads=n_head, 
        n_layers=n_layer, 
        d_ff=n_embd*4, 
        max_seq_length=block_size, 
        dropout=0.1
    ).to(device)

    classifier = SimpleClassifier(n_input, n_hidden, n_output).to(device)

    # 2. Define Optimizer and Loss (Joint Training)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), 
        lr=learning_rate_enco
    )
    criterion = nn.CrossEntropyLoss()

    # 3. Data Loading
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # --- Part 1.3: Joint Training Loop ---
    print("Starting Classifier Training...")
    for epoch in range(epochs_CLS):
        encoder.train()
        classifier.train()
        total_loss = 0

        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            # encoder returns (output, attention_maps)

            embeddings, _ = encoder(xb) 
            
            # Pool embeddings: Mean across the sequence dimension (dim 1)
            pooled_embeddings = embeddings.mean(dim=1) 
        
            logits = classifier(pooled_embeddings)
            loss = criterion(logits, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        '''
        
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1. Create Padding Mask
            mask = (xb != 0).to(device) 
            
            # 2. Forward pass through Encoder
            embeddings, _ = encoder(xb, src_mask=mask) 
            
            # 3. Masked Mean Pooling (Only average the real words)
            mask_expanded = mask.unsqueeze(-1).float() # (B, L, 1)
            sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
            count_tokens = mask_expanded.sum(dim=1).clamp(min=1) # Avoid division by zero
            pooled_embeddings = sum_embeddings / count_tokens 
            
            # 4. Classifier and Loss
            logits = classifier(pooled_embeddings)
            loss = criterion(logits, yb)

            # 5. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        '''
        
        # Part 1.5: Report accuracy after each epoch
        train_acc = compute_classifier_accuracy_custom(encoder, classifier, train_CLS_loader)
        if (epoch + 1) % 15 ==0:
            print(f"Epoch {epoch+1}/{epochs_CLS} | Loss: {total_loss/len(train_CLS_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # --- Part 1.5: Final Evaluation ---
    test_acc = compute_classifier_accuracy_custom(encoder, classifier, test_CLS_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # --- Part 1.4: Sanity Checks ---
    print("\nRunning Sanity Checks...")
    utils = Utilities(tokenizer, encoder)
    sample_sentence = "This is a sample sentence to check the attention maps."
    utils.sanity_check(sample_sentence, block_size)




def main2():
    # ... (keep existing data loading and tokenizer setup)
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    vocab_size = tokenizer.vocab_size
    
    print("Vocabulary size is", vocab_size)

    # --- Part 2.1: Initialize Decoder ---
    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size, 
        n_embd=n_embd, 
        n_head=n_head, 
        n_layer=n_layer, 
        block_size=block_size, 
        dropout=0.1
    ).to(device)

    # Define optimizer for LM task
    optimizer_LM = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # --- Part 2.2: Decoder Pretraining (LM Task) ---
    print("\nStarting Language Model Pretraining (500 iterations)...")
    
    # Load Training Data for LM
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    decoder.train()
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: # Stop at 500 iterations
            break
        
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass: your decoder should return (loss, attn_maps)
        loss, _ = decoder(xb, yb)

        # Backward pass
        optimizer_LM.zero_grad()
        loss.backward()
        optimizer_LM.step()

        # Part 2.4: Report perplexity every 100 iterations
        if i % 100 == 0:
            # Training perplexity on the current batch
            train_perplexity = torch.exp(loss).item()
            print(f"Iteration {i}: Training Loss {loss.item():.4f}, Perplexity {train_perplexity:.2f}")

    # --- Part 2.4: Evaluation on Specific Test Sets ---
    print("\nEvaluating Perplexity on Politician Test Sets...")
    test_files = ["test_LM_obama.txt", "test_LM_hbush.txt", "test_LM_wbush.txt"]
    
    for test_file in test_files:
        filepath = os.path.join("speechesdataset", test_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            test_text = f.read()
        
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Using the provided helper function (ensure it doesn't have the 'total_loss' bug)
        perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
        print(f"Perplexity on {test_file}: {perplexity:.2f}")

    # --- Part 2.3: Sanity Checks ---
    print("\nRunning Decoder Sanity Checks...")
    utils = Utilities(tokenizer, decoder)
    sample_sentence = "The future of our nation depends on the education of our children."
    utils.sanity_check_decoder(sample_sentence, block_size)    



def main3():
    
    window_size = 4
    print('window size:', window_size)
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    vocab_size = tokenizer.vocab_size
    
    print("Vocabulary size is", vocab_size)

    # --- Part 2.1: Initialize Decoder ---
    decoder = SparseDecoder(
        vocab_size=tokenizer.vocab_size, 
        n_embd=n_embd, 
        n_head=n_head, 
        n_layer=n_layer, 
        block_size=block_size, 
        window_size=window_size,
        dropout=0.1
    ).to(device)

    # Define optimizer for LM task
    optimizer_LM = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # --- Part 2.2: Decoder Pretraining (LM Task) ---
    print("\nStarting Language Model Pretraining (500 iterations)...")
    
    # Load Training Data for LM
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    decoder.train()
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: # Stop at 500 iterations
            break
        
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass: your decoder should return (loss, attn_maps)
        loss, _ = decoder(xb, yb)

        # Backward pass
        optimizer_LM.zero_grad()
        loss.backward()
        optimizer_LM.step()

        # Part 2.4: Report perplexity every 100 iterations
        if i % 100 == 0:
            # Training perplexity on the current batch
            train_perplexity = torch.exp(loss).item()
            print(f"Iteration {i}: Training Loss {loss.item():.4f}, Perplexity {train_perplexity:.2f}")

    # --- Part 2.4: Evaluation on Specific Test Sets ---
    print("\nEvaluating Perplexity on Politician Test Sets...")
    test_files = ["test_LM_obama.txt", "test_LM_hbush.txt", "test_LM_wbush.txt"]
    
    for test_file in test_files:
        filepath = os.path.join("speechesdataset", test_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            test_text = f.read()
        
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Using the provided helper function (ensure it doesn't have the 'total_loss' bug)
        perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
        print(f"Perplexity on {test_file}: {perplexity:.2f}")

    # --- Part 2.3: Sanity Checks ---
    print("\nRunning Decoder Sanity Checks...")
    utils = Utilities(tokenizer, decoder)
    sample_sentence = "The future of our nation depends on the education of our children."
    utils.sanity_check_decoder(sample_sentence, block_size)   


if __name__ == "__main__":
    print('running transformer encoder (assignment part 1)')
    main1()
    print('running transformer decoder (assignment part 2)')
    main2()
    print('running transformer decoder with sparse attention (assignment part 3)')
    main3()

