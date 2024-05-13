import os
import pickle
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from langdetect import detect, LangDetectException
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spellchecker import SpellChecker
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split


contractions_dict = {
    "won't": "will not",
    "can't": "cannot",
    "ain't": "is not",
    "aren't": "are not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "let's": "let us",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "what's": "what is",
    "who's": "who is",
    "you're": "you are",
    "I'm": "I am"
}


class CustomerServiceDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        item = {'input_ids': self.inputs['input_ids'][idx], 'attention_mask': self.inputs['attention_mask'][idx],
                'labels': self.labels['input_ids'][idx]}
        return item


def is_english(text):
    try:
        # Returns True if the detected language is English
        return detect(text) == 'en'
    except LangDetectException:
        # If the language detection fails, assume it's not English
        return False


def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('(%s)' % '|'.join(contractions_dict.keys()),
                                      flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match.lower(), match)  # Fallback to match if not found
        # Ensure we handle cases where the expansion isn't found
        if expanded_contraction:
            return first_char + expanded_contraction[1:]
        else:
            return match  # Return the original if no expansion is found

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text


def preprocess_text(text):
    print("Preprocessing data")
    text = expand_contractions(text, contractions_dict)
    text = remove_emojis(text)
    # Remove URLs, user tags, and non-alphanumeric characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Remove non-alphabetic characters except spaces
    text = text.lower()
    if not is_english(text):
        return None
    # Spell correction
    spell = SpellChecker()
    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word:
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)  # Append the original word if correction is None
    return ' '.join(corrected_words)


def remove_emojis(text):
    # Regex to filter out emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def save_checkpoint(model, optimizer, epoch, loss, path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(model, optimizer, path="model_checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def validate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No gradients are needed for validation
        for batch in val_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss


def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No gradients are needed for testing
        for batch in test_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def tokenize_batches(pairs, tokenizer, max_length, batch_size=32):
    # Prepare lists to collect batches
    batched_inputs = {'input_ids': [], 'attention_mask': []}
    batched_labels = {'input_ids': []}

    # Temporary lists to store data for each batch
    temp_inputs = []
    temp_labels = []

    for texts_in, texts_out in pairs:
        # Tokenize without adding to tensors yet
        inp = tokenizer(texts_in, truncation=True, max_length=max_length, padding=False, add_special_tokens=True)
        lbl = tokenizer(texts_out, truncation=True, max_length=max_length, padding=False, add_special_tokens=True)

        temp_inputs.append(inp)
        temp_labels.append(lbl)

        # Check if we've reached the batch size
        if len(temp_inputs) == batch_size:
            # Pad the collected batch and add to the main list
            batch_inputs = tokenizer.pad(temp_inputs, return_tensors="pt", max_length=max_length, padding='max_length')
            batch_labels = tokenizer.pad(temp_labels, return_tensors="pt", max_length=max_length, padding='max_length')

            batched_inputs['input_ids'].append(batch_inputs['input_ids'])
            batched_inputs['attention_mask'].append(batch_inputs['attention_mask'])
            batched_labels['input_ids'].append(batch_labels['input_ids'])

            # Reset temporary lists
            temp_inputs = []
            temp_labels = []

    # Process any remaining pairs after loop
    if temp_inputs:
        batch_inputs = tokenizer.pad(temp_inputs, return_tensors="pt", max_length=max_length, padding='max_length')
        batch_labels = tokenizer.pad(temp_labels, return_tensors="pt", max_length=max_length, padding='max_length')

        batched_inputs['input_ids'].append(batch_inputs['input_ids'])
        batched_inputs['attention_mask'].append(batch_inputs['attention_mask'])
        batched_labels['input_ids'].append(batch_labels['input_ids'])

    # Concatenate all batches into single tensors
    batched_inputs['input_ids'] = torch.cat(batched_inputs['input_ids'], dim=0)
    batched_inputs['attention_mask'] = torch.cat(batched_inputs['attention_mask'], dim=0)
    batched_labels['input_ids'] = torch.cat(batched_labels['input_ids'], dim=0)

    return batched_inputs, batched_labels


def split_data(pairs):
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    print("Data split into training, validation, and testing sets.")
    return train_pairs, val_pairs, test_pairs


def create_datasets(train_pairs, val_pairs, test_pairs, tokenizer, max_length):
    print("Tokenizing data...")
    train_inputs, train_labels = tokenize_batches(train_pairs, tokenizer, max_length)
    val_inputs, val_labels = tokenize_batches(val_pairs, tokenizer, max_length)
    test_inputs, test_labels = tokenize_batches(test_pairs, tokenizer, max_length)

    train_dataset = CustomerServiceDataset(train_inputs, train_labels)
    val_dataset = CustomerServiceDataset(val_inputs, val_labels)
    test_dataset = CustomerServiceDataset(test_inputs, test_labels)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaders created.")
    return train_loader, val_loader, test_loader


def calculate_token_lengths(text, tokenizer):
    return len(tokenizer.encode(text))


def get_max_length(raw_data):
    percentile_90 = raw_data['token_length'].quantile(0.90)
    print(f"The 90th percentile of token lengths is: {percentile_90}")

    # Use this value as max_length in tokenization
    max_length = int(percentile_90)
    return max_length


def process_raw_data(raw_data_filepath, tokenizer):
    print("Processing raw data")
    raw_data = pd.read_csv(raw_data_filepath)
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(preprocess_text, raw_data['text']))
    raw_data['processed_text'] = results
    raw_data.dropna(subset=['processed_text'], inplace=True)
    raw_data = process_token_lengths(raw_data, tokenizer)
    return raw_data


def process_token_lengths(raw_data, tokenizer):
    raw_data['token_length'] = raw_data['processed_text'].apply(
        lambda x: calculate_token_lengths(x, tokenizer) if pd.notna(x) else 0)
    return raw_data


def convert_to_qa(raw_data):
    # Merge the data to form pairs
    conversations = raw_data[raw_data['inbound']].merge(
        raw_data[raw_data['inbound'] == False],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id'
    )

    # Create pairs and filter out any with NaN values
    pairs = [(text_in, text_out) for text_in, text_out in
             zip(conversations['processed_text_x'], conversations['processed_text_y'])
             if pd.notna(text_in) and pd.notna(text_out)]

    return pairs


def main():
    raw_data_filepath = 'twcs/twcs.csv'
    preprocessed_data_filepath = 'preprocessed_data.csv'
    preprocessed_pkl_filepath = 'preprocessed_data.pkl'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(preprocessed_pkl_filepath):
        if not os.path.exists(preprocessed_data_filepath):
            raw_data = process_raw_data(raw_data_filepath, tokenizer)
            raw_data.to_csv(preprocessed_data_filepath, index=False)
        raw_data = pd.read_csv(preprocessed_data_filepath)
        raw_data = process_token_lengths(raw_data, tokenizer)
        max_length = get_max_length(raw_data)
        print(f"The maximum length of text is: {max_length}")
        pairs = convert_to_qa(raw_data)
        save_data(pairs, preprocessed_pkl_filepath)

    pairs = load_data(preprocessed_pkl_filepath)
    train_pairs, val_pairs, test_pairs = split_data(pairs)
    train_dataset, val_dataset, test_dataset = create_datasets(train_pairs, val_pairs, test_pairs, tokenizer,
                                                               30)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=300)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    epoch_start, _ = load_checkpoint(model, optimizer) if os.path.exists('model_checkpoint.pth') else (0, float('inf'))
    num_epochs = 5
    for epoch in range(epoch_start, num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}')

        # Validation phase
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        # Testing phase
        test_loss = test(model, test_loader, device)

        # Output results
        print(f'Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader):.4f}, '
              f'Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

        save_checkpoint(model, optimizer, epoch + 1, total_loss / len(train_loader))

    # Save final model
    if isinstance(model, DataParallel):
        model.module.save_pretrained('model_final')
    else:
        model.save_pretrained('model_final')
    tokenizer.save_pretrained('model_final')


if __name__ == "__main__":
    main()
