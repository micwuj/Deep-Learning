import pandas as pd
from transformers import pipeline, AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

def load_data(dev_x_path, dev_y_path):
    dev_x = pd.read_csv(dev_x_path, sep="\t", names=["text"])
    dev_y = pd.read_csv(dev_y_path, sep="\t", names=["label"])
    dev_x["tokens"] = dev_x["text"].apply(lambda x: x.split())
    dev_y["labels"] = dev_y["label"].apply(lambda x: x.split())
    return dev_x["tokens"].tolist(), dev_y["labels"].tolist()

def predict_labels(tokens_list, ner_pipe, tokenizer):
    predicted_labels = []

    for tokens in tqdm(tokens_list, desc="Predicting"):
        inputs = tokenizer(tokens,
                           is_split_into_words=True,
                           return_offsets_mapping=True,
                           return_tensors="pt",
                           truncation=True)

        word_ids = inputs.word_ids()
        ner_results = ner_pipe(" ".join(tokens))

        token_tags = ["O"] * len(tokens)

        for entity in ner_results:
            idx = entity["index"] - 1
            if idx < len(word_ids):
                word_id = word_ids[idx]
                if word_id is not None:
                    token_tags[word_id] = entity["entity"]

        predicted_labels.append(token_tags)

    return predicted_labels

def main():
    dev_x_path = "en-ner-conll-2003/dev-0/in.tsv"
    dev_y_path = "en-ner-conll-2003/dev-0/expected.tsv"

    tokens_list, true_labels = load_data(dev_x_path, dev_y_path)

    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipe = pipeline("ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=None)

    predicted_labels = predict_labels(tokens_list, ner_pipe, tokenizer)

    print("F1 score:", f1_score(true_labels, predicted_labels))
    print("\nClassification report:")
    print(classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    main()
