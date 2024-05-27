import torch
from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
from argparse import Namespace
from models.fc_net import FCNet
from data.tabular_data_module import TabularDataModule

LR = 1e-5
DATA_CONFIG = {'input_shape': torch.Size([9264, 501]), 'output_dims': (1,), 'mapping': ['0', '1'], 'batch_size': 32}
FC_DIMS = [2048, 1024, 1024, 256]

# define the LightningModule
class LitTabularModel(L.LightningModule):
    def __init__(self, data_config: dict = DATA_CONFIG, model: str = 'FCNet', 
                 lr: float = LR, fc_dims = FC_DIMS):
        """ LightningModule model for tabular data
        Parameters
        ----------
        data_config
            Configuration of the data (input width, batch size, etc).
        model
            Pytorch model to train.
        lr
            Learning rate.
        """
        super().__init__()
        # params
        self.data_config = data_config
        batch_size = self.data_config['batch_size']
        self.lr = lr
        
        # model
        if model == 'FCNet':
            model = FCNet(data_config, fc_dims)
        self.model = model
        input_width = self.model.input_width
        
        # loss + metrics
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        # use this to show input dimensions of the models
        self.example_input_array = torch.Tensor(batch_size, input_width)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # Log accuracy metric
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)
        # Log accuracy metric
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        return self.model(x)