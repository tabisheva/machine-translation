import torch
from torchtext.data import TabularDataset, Field
from torchtext.data.iterator import BucketIterator
from transformer import Transformer
from utils import prepare_csv, ScheduledOpt, init_weights, translate_sentence_beam, compute_bleu_score
from config_translate import params
import torch.nn as nn
import time
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(24)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(24)

EN_TEXT = Field(init_token = "<sos>", eos_token = "<eos>", batch_first = True)
DE_TEXT = Field(init_token = '<sos>', eos_token = '<eos>', batch_first = True)

prepare_csv(params["train_source_path"],
            params["train_target_path"],
            'train.csv')
prepare_csv(params["val_source_path"],
            params["val_target_path"],
            'val.csv')

data_fields = [('src', DE_TEXT), ('trg', EN_TEXT)]
train, val = TabularDataset.splits(path='./',
                                   train='train.csv',
                                   validation='val.csv',
                                   format='csv',
                                   fields=data_fields)

DE_TEXT.build_vocab(train, val, min_freq = params["min_freq"])
EN_TEXT.build_vocab(train, val, min_freq = params["min_freq"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = BucketIterator(train, batch_size=params["batch_size"],
                            sort_key=lambda x: len(x.src),
                            shuffle=True,
                            device=device,
                            sort_within_batch=True)
val_iter = BucketIterator(val, batch_size=params["batch_size"],
                          sort_key=lambda x: len(x.src),
                          shuffle=False,
                          device=device,
                          sort_within_batch=True)

INPUT_DIM = len(DE_TEXT.vocab)
OUTPUT_DIM = len(EN_TEXT.vocab)

model = Transformer(INPUT_DIM, OUTPUT_DIM, device).to(device)
model.apply(init_weights)

optimizer = ScheduledOpt(params["d_model"], params["warmup_steps"],
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

criterion = nn.CrossEntropyLoss(ignore_index = 1)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
        optimizer.step()
        optimizer.optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            bleu_score = compute_bleu_score(output, trg, EN_TEXT)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), bleu_score

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_val_loss = 10.0

for epoch in range(params["n_epochs"]):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion)
    val_loss = train(model, val_iter, optimizer, criterion)
    valid_loss, bleu_score = evaluate(model, val_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), 'translate_model.pt')
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} \t Val bleu: {bleu_score:.3f}')

model.load_state_dict(torch.load('translate_model.pt'))
model.eval()
f_in = open(params["test_path"], 'r', encoding='utf-8')
f_out = open(params["output_path"], 'w', encoding='utf-8')
for line in f_in:
    translation = translate_sentence_beam(line, DE_TEXT, EN_TEXT, model, device)
    translation = ' '.join(translation) + '\n'
    f_out.write(translation)
model = model.cpu()
f_in.close()
f_out.close()
