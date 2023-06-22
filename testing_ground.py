from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
class SimpleLinear(nn.Module):
  def __init__(self, start_shape, hidden_units, end_shape):
    super().__init__()
    self.linear_sequence = nn.Sequential(
      nn.Linear(start_shape, hidden_units),
      nn.Linear(hidden_units, end_shape)
    )
  def forward(self,x):
    return self.linear_sequence(x)


class SimpleLSTM(nn.Module):
  def __init__(self, start_shape, hidden_units, end_shape, num_stack_layers):
    super().__init__()
    self.num_stacked_layers = num_stack_layers
    self.hidden_units = hidden_units
    self.lstm = nn.LSTM(start_shape, hidden_units, num_stack_layers, batch_first=True)
    self.finish = nn.Linear(hidden_units, end_shape)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  def forward(self, x):
    h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_units).to(self.device)
    c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_units).to(self.device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.finish(out[:, -1, :])
    return out

from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
class train_regression_model():
    def __init__(self, model: nn.Module, data_loader, optimizer: torch.optim, loss_fn, epochs: int, ):
        for epoch in tqdm(range(epochs)):
            # print(f"Epoch: {epoch}")
            train_loss = 0
            train_acc = 0
            for batch, (X, y) in enumerate(data_loader):
                model.train()
                y_preds = model(X)

                loss = loss_fn(y_preds, y)
                train_loss += loss
                train_acc += mean_absolute_error(y.cpu().detach().numpy(),
                                                 y_preds.cpu().detach().numpy())
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            train_loss /= len(data_loader)
            train_acc /= len(data_loader)

            print(f"\nTrain loss: {train_loss} | Train acc: {train_acc}")

class evaluate_regression_model():
    def __init__(self, model: nn.Module, data_loader,loss_fn):
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for X, y in data_loader:
                self.test_pred = model(X)
                loss = loss_fn(self.test_pred, y)
                test_loss += loss

                test_acc += mean_absolute_error(y.cpu().detach().numpy(),
                                                self.test_pred.cpu().detach().numpy())

            test_loss /= len(data_loader)
            test_acc /= len(data_loader)

        print(f"Test loss: {test_loss} | Test acc: {test_acc}\n")

    def get_prediction(self):
        return self.test_pred


class CustomDataSet(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return X_train[idx], y_train[idx]

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_prediction(train_data = X_train,
                    train_labels = y_train,
                    test_data = X_test,
                    test_labels = y_test,
                    prediction=None):
  plt.figure(figsize=(10,7))
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

  if prediction is not None:
    plt.scatter(test_data, prediction, c="r", s=4, label="Prediction")

  plt.legend(prop={"size":14})
  plt.show()

plot_prediction()

model_0 = SimpleLinear(1, 8, 1)

with torch.inference_mode():
  y_preds = model_0(X_test)

plot_prediction(prediction=y_preds) #random prediction without training
X_test[0], y_preds[0]

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)

train_data_set = CustomDataSet(X_train, y_train)
train_regression_model(model_0, train_data_set, optimizer, loss_fn, 100)
test_data_set = CustomDataSet(X_test, y_test)
prediction = evaluate_regression_model(model_0, test_data_set, loss_fn)
prediction1 = prediction.get_prediction()
plot_prediction(prediction=prediction1)


