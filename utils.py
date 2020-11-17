import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu


class ScheduledOpt:

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # https://arxiv.org/pdf/1706.03762.pdf
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def prepare_csv(path_de, path_en=None, out_file='train.csv'):
    en = open(path_en, encoding='utf-8').read().split('\n')
    de = open(path_de, encoding='utf-8').read().split('\n')
    raw_data = {'src': [line for line in de], 'trg': [line for line in en]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df['src'].str.count(' ') < 48) & (df['trg'].str.count(' ') < 48)
    df = df.loc[mask]
    df.to_csv(out_file, index=False)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def translate_sentence_beam(sentence, src_field, trg_field, model, device, width=3):
    tokens = [token.lower() for token in sentence.split(' ')]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    max_len = len(tokens)
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_enc, src_mask, _ = model.get_masks(src_tensor)
    with torch.no_grad():
        enc_src = model.transformer.encoder(src_enc)
    trg_tensor = torch.zeros(max_len).type_as(src_tensor.data)
    trg_tensor[0] = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]])
    beam_pool = [(trg_tensor, 1.0)]
    for i in range(1, max_len):
        beam_history = []
        for beam, old_prob in beam_pool:
            outputs_enc, tgt_mask, tgt_pad_mask = model.get_masks(beam[:i].unsqueeze(0))
            with torch.no_grad():
                output = model.transformer.decoder(outputs_enc, enc_src, tgt_mask=tgt_mask,
                                                   tgt_key_padding_mask=tgt_pad_mask)
            out = output.transpose(0, 1)
            output = model.out(out).cpu().detach()
            out = F.softmax(output, dim=-1)
            probs, ix = out[:, -1].topk(width)
            for prob, token_id in zip(probs.squeeze(), ix.squeeze()):
                tmp_beam = beam.clone()
                tmp_beam[i] = token_id.item()
                beam_history.append((tmp_beam, prob * old_prob))
        beam_pool = sorted(beam_history, key=lambda x: x[1], reverse=True)[:width]
        if beam_pool[0][0][i] == 3:
            break
    best_beam = beam_pool[0][0]
    best_beam = best_beam[best_beam != 0]
    i = (best_beam == 3).nonzero()[0] + 1 if 3 in best_beam else best_beam.shape[0]
    trg_tokens = [trg_field.vocab.itos[i] for i in best_beam[:i]]
    return trg_tokens[1:]


def compute_bleu_score(outputs, targets, target_field):
    refs = []
    targets = targets.tolist()
    for target in targets:
        refs.append(' '.join([target_field.vocab.itos[i] for i in target if (i != 2 and i != 3 and i != 1)]))
    refs = [refs]
    preds = []
    pred_indexes = outputs.argmax(2).tolist()
    for pred in pred_indexes:
        pred_tokens = [target_field.vocab.itos[i] for i in pred if (i != 1 and i != 2 and i != 3)]
        preds.append(' '.join(pred_tokens))
    bleu = sacrebleu.corpus_bleu(preds, refs).score
    return bleu
