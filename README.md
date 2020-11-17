# Machine Translation

Neural machine translation using Transformer model from Pytorch.

## Usage

Install `requirements.txt`

Set paths to train/val/test files in `config_translate.py`

Run `python main.py`

## Results

The best model will be saved  as `translate_model.pt`, translated test file - `test1.de-en.en`.

Bleu score of translation from German to English is `28.3`
