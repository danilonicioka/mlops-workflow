import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import kfp
from kfp import dsl

class InterruptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=29, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

@dsl.component
def train_and_evaluate(
    epochs: int,
    learning_rate: float,
    X_train_path: str,
    y_train_path: str,
    X_test_path: str,
    y_test_path: str,
    model_output_path: str,
    metrics_output_path: str
):
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    X_train = torch.tensor(np.load(X_train_path), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.load(y_train_path), dtype=torch.float32).to(device)
    X_test = torch.tensor(np.load(X_test_path), dtype=torch.float32).to(device)
    y_test = torch.tensor(np.load(y_test_path), dtype=torch.float32).to(device)

    model = InterruptionModel().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    model.eval()
    with torch.no_grad():
        y_preds = torch.round(torch.sigmoid(model(X_test))).squeeze()

    predictions = y_preds.cpu().numpy()
    true_labels = y_test.cpu().numpy()

    metrics = {
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, average='weighted'),
        "recall": recall_score(true_labels, predictions, average='weighted'),
        "micro_f1": f1_score(true_labels, predictions, average='micro'),
        "macro_f1": f1_score(true_labels, predictions, average='macro'),
        "classification_report": classification_report(true_labels, predictions, target_names=['No-Stall', 'Stall'])
    }

    # Save model
    torch.save(model.state_dict(), model_output_path)

    # Save metrics
    with open(metrics_output_path, 'w') as f:
        f.write(str(metrics))

# Define the pipeline
@dsl.pipeline(
    name='Interruption Model Training Pipeline',
    description='A pipeline to train and evaluate an interruption model using PyTorch and KFP.'
)
def interruption_training_pipeline(
    epochs: int,
    learning_rate: float,
    X_train_path: str,
    y_train_path: str,
    X_test_path: str,
    y_test_path: str,
    model_output_path: str,
    metrics_output_path: str
):
    train_and_evaluate(
        epochs=epochs,
        learning_rate=learning_rate,
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_test_path=X_test_path,
        y_test_path=y_test_path,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path
    )

# Compile the pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(interruption_training_pipeline, 'interruption_training_pipeline.yaml')
