import kfp
from kfp import dsl
import torch

@dsl.component(base_image="python:3.12.3", packages_to_install=['torch', 'scikit-learn'])
def modeltraining(
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    X_test: torch.Tensor, 
    y_test: torch.Tensor,
    lr: float = 0.0001, 
    epochs: int = 3500, 
    seed: int = 42, 
    print_every: int = 500
    ) -> str:
    
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model with non-linear activation function
    class InterruptionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=29, out_features=200)
            self.layer_2 = nn.Linear(in_features=200, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
            self.relu = nn.ReLU() # <- add in ReLU activation function
            # Can also put sigmoid in the model
            # This would mean you don't need to use it on the predictions
            # self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Intersperse the ReLU activation function between layers
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    model_3 = InterruptionModel().to(device)
    print(model_3)

    # Setup loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_3.parameters(), lr=lr)

    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc

    # Fit the model
    torch.manual_seed(seed)

    # Assuming X_train, y_train, X_test, y_test are already defined and are tensors
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        # 1. Forward pass
        #model_3.train()
        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model_3.eval()
        with torch.no_grad():
            # 1. Forward pass
            test_logits = model_3(X_test).squeeze()
            #print(test_logits.shape)
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels

            # 2. Calculate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print out what's happening
        if epoch % print_every == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    # Evaluate the final model
    model_3.eval()
    with torch.no_grad():
        y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

    predictions = y_preds.cpu().numpy() # if using cuda, otherwise y_pred.numpy()
    true_labels = y_test.cpu().numpy()

    print("=== Confusion Matrix ===")
    print(confusion_matrix(true_labels, predictions))
    print('\n')

    print("=== Score ===")
    accuracy = accuracy_score(true_labels, predictions)
    print('Accuracy: %f' % accuracy)

    precision = precision_score(true_labels, predictions, average='weighted')
    print('Precision: %f' % precision)
    recall = recall_score(true_labels, predictions, average='weighted')
    print('Recall: %f' % recall)

    microf1 = f1_score(true_labels, predictions, average='micro')
    print('Micro F1 score: %f' % microf1)
    macrof1 = f1_score(true_labels, predictions, average='macro')
    print('Macro F1 score: %f' % macrof1)

    target_names = ['No-Stall', 'Stall']

    # Print precision-recall report
    print(classification_report(true_labels, predictions, target_names=target_names))
