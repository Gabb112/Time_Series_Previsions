import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from models.model_trainer import ModelTrainer
import copy


class LSTMModel(ModelTrainer):
    """LSTM model for forecasting time series."""

    def __init__(
        self,
        input_size=1,
        hidden_size=50,
        num_layers=1,
        output_size=1,
        learning_rate=0.001,
        num_epochs=100,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None
        self.scaler = MinMaxScaler()
        self.trained = False

    def _create_sequences(self, data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i : (i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train(self, train_data, val_data, target_column="Adj Close", seq_length=10):
        """Trains the LSTM model."""
        if train_data is None or target_column not in train_data.columns:
            print("Error: Train data is invalid, can't train.")
            return None
        if val_data is None or target_column not in val_data.columns:
            print("Error: Validation data is invalid, can't train.")
            return None
        try:
            train_data_values = train_data[target_column].values.reshape(-1, 1)
            val_data_values = val_data[target_column].values.reshape(-1, 1)
            # Normalizing the train and validation data
            train_data_scaled = self.scaler.fit_transform(train_data_values)
            val_data_scaled = self.scaler.transform(val_data_values)

            # Create sequences
            train_seq, train_labels = self._create_sequences(
                train_data_scaled, seq_length
            )
            val_seq, val_labels = self._create_sequences(val_data_scaled, seq_length)

            # Convert data to tensors and create dataloaders
            train_seq_tensor = torch.FloatTensor(train_seq)
            train_labels_tensor = torch.FloatTensor(train_labels)
            val_seq_tensor = torch.FloatTensor(val_seq)
            val_labels_tensor = torch.FloatTensor(val_labels)

            train_dataset = TensorDataset(train_seq_tensor, train_labels_tensor)
            val_dataset = TensorDataset(val_seq_tensor, val_labels_tensor)

            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Define and initialize the model

            class SequenceModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers, batch_first=True
                    )
                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    h0 = torch.zeros(
                        self.num_layers, x.size(0), self.hidden_size
                    ).requires_grad_()
                    c0 = torch.zeros(
                        self.num_layers, x.size(0), self.hidden_size
                    ).requires_grad_()
                    out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
                    out = self.fc(out[:, -1, :])  # Use the last output of sequence
                    return out

            self.model = SequenceModel(
                self.input_size, self.hidden_size, self.num_layers, self.output_size
            )

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            best_val_loss = float("inf")
            best_model_wts = copy.deepcopy(self.model.state_dict())

            # Training loop
            for epoch in range(self.num_epochs):
                self.model.train()
                for sequences, labels in train_dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Validation step
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for val_sequences, val_labels in val_dataloader:
                        val_outputs = self.model(val_sequences)
                        loss = criterion(val_outputs, val_labels)
                        val_loss += loss.item()
                    val_loss /= len(val_dataloader)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                    print(
                        f"Epoch: {epoch+1}/{self.num_epochs}, Training Loss:{loss.item():.4f}, Validation Loss: {val_loss:.4f}"
                    )

            self.model.load_state_dict(best_model_wts)
            self.trained = True
            print("LSTM model has been trained.")
            return self
        except Exception as e:
            print(f"Error during LSTM model training: {e}")
            return None

    def predict(self, test_data, target_column="Adj Close", seq_length=10):
        """Makes predictions using the trained LSTM model."""
        if not self.trained:
            print("Error: Model is not trained. Please train before using predict.")
            return None
        if test_data is None or target_column not in test_data.columns:
            print("Error: Test data is invalid, cannot make predictions")
            return None

        try:
            test_data_values = test_data[target_column].values.reshape(-1, 1)
            test_data_scaled = self.scaler.transform(test_data_values)
            test_seq, _ = self._create_sequences(test_data_scaled, seq_length)
            test_seq_tensor = torch.FloatTensor(test_seq)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(test_seq_tensor).numpy()

            # Inverse transform of the predictions
            predictions = self.scaler.inverse_transform(predictions)
            return predictions.flatten()
        except Exception as e:
            print(f"Error during LSTM model prediction: {e}")
            return None
