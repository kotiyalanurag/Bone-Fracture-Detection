import os
import mlflow.pytorch
import torch.optim.lr_scheduler as lr_scheduler

from utils import *
from train_helper import *
from model import CustomResnet

def main(device, train_data, val_data, epochs = 3, lr = 0.001, gamma = 0.99):
    
    """ A function to train our model

    Args:
        device (device): mps or cpu
        train_data (torch.utils.data.DataLoader): train dataloader
        val_data (torch.utils.data.DataLoader): val dataloader
        epochs (int, optional): Number of epochs to train our model. Defaults to 3.
        lr (float, optional): Initial learning rate for our model. Defaults to 0.001.
        gamma (float, optional): LR decay rate for model. Defaults to 0.99 (reduces lr by 1% on every 3rd epoch).
    """
    
    model = CustomResnet(2).to(device)  # initialising model
    
    loss_fn = torch.nn.CrossEntropyLoss()   # setting up a loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)    # setting up an optimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = gamma)     # setting up a learning rate scheduler

    tl = [] # stores training loss over epochs
    ta = [] # stores training accuracy over epochs
    vl = [] # stores validation loss over epochs
    va = [] # stores validation accuracy over epochs

    # best_acc = 0.0

    os.makedirs('Models', exist_ok = True) # making a Models directory to save trained model

    print(f'\nModel training started at {timestamp()}\n')
    for epoch in range(epochs):
        
        print(f"\nEpoch {epoch+1}/{epochs} started at {time()}")
        print('=' * 80)    
        
        model.train()
        avg_train_loss, train_accuracy = train_epoch(model, device, train_data, 
                                                     train_image_paths, optimizer, 
                                                     loss_fn)
        
        model.eval()
        avg_val_loss, val_accuracy = val_epoch(model, device, val_data, 
                                               val_image_paths, loss_fn)  
                                               
        tl.append(avg_train_loss)
        ta.append(train_accuracy)
        vl.append(avg_val_loss)
        va.append(val_accuracy)
        
        print(f"Training loss: {avg_train_loss:.2f} || Training accuracy: {train_accuracy:.2f}")
        print(f"Validation loss: {avg_val_loss:.2f} || Validation accuracy: {val_accuracy:.2f}")
        
        if (epoch + 3) % 2 == 0:
                
            scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]

            print(f'Learning rate changed to {new_lr:.6f}')
            
        # if best_acc < val_accuracy:
        #     best_acc = val_accuracy
        #     save_model(model, path = "Models/customresnet_best_weights.pt")  
    
    # tracking experiments with mlflow 
    mlflow.pytorch.autolog(disable = True)
    
    with mlflow.start_run(run_name = "Resnet50"):
        
        mlflow.set_tag("model_name", "Custom Resnet")
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning rate", lr)
        mlflow.log_param("scheduler rate", gamma)
        
        avg_test_loss, test_accuracy = test_epoch(model, device, test_data, test_image_paths, loss_fn)
        
        mlflow.log_metric("test accuracy", test_accuracy)
        mlflow.log_metric("test loss", avg_test_loss)
        
if __name__ == "__main__":
    
    main(device, train_data, val_data)