from src.data_processing import read_sentiment_examples, build_vocab, bag_of_words
from src.naive_bayes import NaiveBayes
from src.logistic_regression import LogisticRegression
from src.utils import evaluate_classification
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training data
    train_examples = read_sentiment_examples("data/train.txt")
    print("Building vocabulary...")
    vocab = build_vocab(train_examples)

    # Prepare features and labels for the models
    print("Preparing training BoW...")
    train_features = torch.stack(
        [bag_of_words(ex.words, vocab) for ex in train_examples]
    ).to(device, dtype=torch.float32)
    train_labels = torch.tensor(
        [ex.label for ex in train_examples], dtype=torch.float32
    ).to(device)

    print("Training Naive Bayes model...")
    # Train Naive Bayes model
    nb_model = NaiveBayes()
    nb_model.fit(train_features, train_labels)

    print("Training Logistic Regression model...")
    # Train Logistic Regression model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(train_features, train_labels, learning_rate=0.1, epochs=1000)

    # Load test data
    test_examples = read_sentiment_examples("data/test.txt")
    print("Preparing test BoW...")
    test_features = torch.stack([bag_of_words(ex.words, vocab) for ex in test_examples]).to(device)
    test_labels = torch.tensor([ex.label for ex in test_examples], dtype=torch.float32).to(device)

    # Evaluate Naive Bayes model
    nb_predictions = [nb_model.predict(ex.to(device)) for ex in test_features]
    nb_metrics = evaluate_classification(torch.tensor(nb_predictions, device=device), test_labels)
    print("Naive Bayes Metrics:", nb_metrics)

    # Evaluate Logistic Regression model
    lr_predictions = lr_model.predict(test_features)
    lr_metrics = evaluate_classification(lr_predictions, test_labels)
    print("Logistic Regression Metrics:", lr_metrics)


if __name__ == "__main__":
    main()
