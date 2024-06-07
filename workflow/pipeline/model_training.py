from kfp.dsl import Input, Output, Dataset, Model, component, ClassificationMetrics, Metrics

@component(base_image="python:3.11.9", packages_to_install=['torch==2.3.0', 'scikit-learn==1.2.2'])
def model_training(
    X_train_artifact: Input[Dataset], 
    X_test_artifact: Input[Dataset],
    y_train_artifact: Input[Dataset],
    y_test_artifact: Input[Dataset],
    metrics: Output[Metrics], 
    classification_metrics: Output[ClassificationMetrics], 
    model_trained_artifact: Output[Model],
    lr: float = 0.0001,
    epochs: int = 3500,
    print_every: int = 500
    ):
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InterruptionModel().to(device)

    # Setup loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc
    
    # Fit the model
    torch.manual_seed(42)
    epochs = epochs

    # Put all data on target device
    X_train = torch.load(X_train_artifact.path)
    X_test = torch.load(X_test_artifact.path)
    y_train = torch.load(y_train_artifact.path)
    y_test = torch.load(y_test_artifact.path)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model(X_train).squeeze()

        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train,
                        y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.no_grad():
        # 1. Forward pass
            test_logits = model(X_test).squeeze()
            #print(test_logits.shape)
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calcuate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)


        # Print out what's happening
        if epoch % print_every == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

        model.eval()
        with torch.no_grad():
            y_preds = torch.round(torch.sigmoid(model(X_test))).squeeze()

        if device == "cuda":
            predictions = y_preds.cpu().numpy() #if it is cuda, then this, otherwise y_pred.numpy()
            true_labels = y_test.cpu().numpy()
        else:
            predictions = y_preds.numpy()
            true_labels = y_test.numpy()
        
        # Confusion Matrix
        cmatrix = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:", cmatrix)

        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        metrics.log_metric("Accuracy", accuracy)
        print('Accuracy: %f' % accuracy)

        precision = precision_score(true_labels,  predictions, average='weighted')
        metrics.log_metric("Precision", precision)
        print('Precision: %f' % precision)

        recall = recall_score(true_labels, predictions, average='weighted')
        metrics.log_metric("Recall", recall)
        print('Recall: %f' % recall)

        microf1 = f1_score(true_labels, predictions, average='micro')
        metrics.log_metric("Micro F1 score", microf1)
        print('Micro F1 score: %f' % microf1)

        macrof1 = f1_score(true_labels, predictions, average='macro')
        metrics.log_metric("Macro F1 score", macrof1)
        print('Macro F1 score: %f' % macrof1)

        target_names = ['No-Stall', 'Stall']
        # Print precision-recall report
        print(classification_report(true_labels, predictions, target_names=target_names))

        # Classification Metrics artifact
        cmatrix = cmatrix.tolist()
        target_names = ['No-Stall', 'Stall']
        classification_metrics.log_confusion_matrix(target_names, cmatrix)

        # Save model
        torch.save(model.state_dict(), model_trained_artifact.path)