import  torch
from tqdm.auto import tqdm
from typing import Dict,Tuple,List

# train step ############################################
def Train_step(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    device:torch.device
)-> Tuple[float , float]:

    model.train()
    train_loss , train_acc = 0,0
    for batch,(x,y) in enumerate(dataloader):
      x,y = x.to(device) , y.to(device)
      #predction
      y_pred = model(x)
      # loss finding
      train_loss += model(x)
      #accuracy finding
      acc+= accuracy_fn(y,y_pred.argmax(dim=1))
      #optimizer
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss , train_acc

# test step #################################################
def Test_step(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device:torch.device
)-> Tuple[float,float]:
    # putting model into the evalution stage
    model.eval()
    test_loss , test_acc = 0,0
    #model testing
    with torch.inferance_mode():
      for batch,(x,y) in enumerate(dataloader):
        x,y = x.to(device) , y.to(device)
        #prediction
        y_pred = model(x)
        test_loss += loss_fn(y_pred , y)
        test_acc += accuracy_fn(y,y_pred.argmax(dim=1))
      test_loss = test_loss/len(dataloader)
      test_acc = test_acc/len(dataloader)
      return test_loss , test_acc

#use teh trainand test loop for each epoch #####################
def train(
    epochs:int,
    model:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    test_dataloader:torch.utils.data.DataLoader,
    optimizer:torch.optim.Optimizer,
    loss_fn:torch.nn.Module,
    device:torch.device
)-> Dict[str , List]:

  result = {
      "train_loss":[] ,
      "train_acc":[],
      "test_loss":[],
      "test_acc":[]
  }
  for epoch in tqdm(range(epochs)):
    train_loss , train_acc = Train_step(model , train_dataloader , loss_fn  , optimizer , device)
    test_loss , test_acc = Test_step(model , test_dataloader , loss_fn , device)
    result["test_acc"].append(test_acc)
    result["test_loss"].append(test_loss)
    result["train_acc"].append(train_acc)
    result["train_loss"].append(train_loss)


    # saving to check the model performance here
    writer.add_scalars("Loss" , tag_scalar_dict={"train_loss":train_loss , "test_loss":test_loss} , global_step=epoch)
    writer.add_scalars("Accuracy" , tag_scalar_dict={"train_acc":train_acc , "test_acc":test_acc} , global_step=epoch)
  return result
