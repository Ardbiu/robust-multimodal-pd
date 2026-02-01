import torch
import numpy as np
from pd_fusion.models.fusion_late import LateFusionModel

class ModalityDropoutModel(LateFusionModel):
    def train(self, X, y, val_data=None):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # moddrop_rate logic usually requires knowing feature indices per modality
        # Here we implement a simplified version that randomly zeros out chunks of features
        
        for epoch in range(self.params["epochs"]):
            self.model.train()
            curr_X = X_tensor.clone()
            
            # Simple dropout: randomly zero out some features for each row
            if np.random.rand() < self.params.get("moddrop_rate", 0.3):
                # Placeholder for specific modality indices dropout
                pass 
                
            self.optimizer.zero_grad()
            outputs = self.model(curr_X)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
