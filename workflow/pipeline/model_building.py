from kfp.dsl import Output, Model, component

@component(base_image="python:3.11.9", packages_to_install=['torch==2.3.0', 'scikit-learn==1.2.2'])
def model_building(
    model_artifact: Output[Model]
    ):
    import torch
    from torch import nn

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

    model = InterruptionModel()

    # Save model
    torch.save(model.state_dict(), model_artifact.path)