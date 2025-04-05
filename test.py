import torch
from transformers import BertForMaskedLM, BertTokenizerFast

import torch
from transformers import DebertaForMaskedLM, DebertaTokenizerFast

def load_model(model_path):
    model = BertForMaskedLM.from_pretrained(model_path)
    return model

def create_tokenizer(pretrained_model_name):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
    special_tokens_dict = {'additional_special_tokens': ['[SYN]', '[EOSYN]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def get_synonyms(text, position, model, tokenizer, k=100):
    # Tokenize input text
    text_list = text.split()
    text_list = text_list[:position] + ['[SYN]'] +[text_list[position]] + ['[EOSYN]'] + text_list[position:]
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    # text_list = text_list[:position] + ['[SYN]'] +[text_list[position]] + ['[EOSYN]'] + text_list[position:]
    # inputs = tokenizer(text_list, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted token ids at the specified position
    pred_token_ids = torch.topk(outputs.logits[0, position + 1], k).indices

    # Convert token ids to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids)
    
    return predicted_tokens

# Load trained model
model_path = ''  # Replace with your model path
model = load_model(model_path)

# Create tokenizer
pretrained_model_name = ''  # Replace with the name of the model you used for training
tokenizer = create_tokenizer(pretrained_model_name)
position = 7
# Specify text and position
text = 'while on a vacation at the beach red haired brothers michael mcgreevey'

# Get synonyms for the word at the specified position
synonyms = get_synonyms(text, position, model, tokenizer)
print(f'Top {len(synonyms)} synonyms for the word at position {position} in the text: "{text}" are: {synonyms}')

# def load_model_Deberta(model_path):
#     model = DebertaForMaskedLM.from_pretrained(model_path)
#     return model
#
# def create_tokenizer_Deberta(pretrained_model_name):
#     tokenizer = DebertaTokenizerFast.from_pretrained(pretrained_model_name)
#     special_tokens_dict = {'additional_special_tokens': ['[SYN]', '[EOSYN]']}
#     tokenizer.add_special_tokens(special_tokens_dict)
#     return tokenizer
#
# def get_synonyms_Deberta(text, position, model, tokenizer, k=5):
#     # Tokenize input text
#     text_list = text.split()
#     text_list = text_list[:position] + ['[SYN]'] +[text_list[position]] + ['[EOSYN]'] + text_list[position:]
#     inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
#
#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     # Get predicted token ids at the specified position
#     pred_token_ids = torch.topk(outputs.logits[0, position + 1], k).indices
#
#     # Convert token ids to tokens
#     predicted_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids)
#
#     return predicted_tokens
#
# # Load trained model
# model_path_Deberta = ''  # Replace with your model path
# model_Deberta = load_model_Deberta(model_path_Deberta)
#
# # Create tokenizer
# pretrained_model_name_Deberta = 'deberta-base-fix'  # Replace with the name of the model you used for training
# tokenizer_Deberta = create_tokenizer_Deberta(pretrained_model_name_Deberta)
# position_Deberta = 10
# # Specify text and position
# text_Deberta = 'after all these years i still consider this series the finest example of world war ii '
#
# # Get synonyms for the word at the specified position
# synonyms_Deberta = get_synonyms_Deberta(text_Deberta, position_Deberta, model_Deberta, tokenizer_Deberta)
# print(f'Top {len(synonyms_Deberta)} synonyms for the word at position {position_Deberta} in the text: "{text_Deberta}" are: {synonyms_Deberta}')