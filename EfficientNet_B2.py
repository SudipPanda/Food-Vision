
from torch import nn
from torchvision import transforms
from torchsummary import summary

def transferlearning (class_num : int ):
  weights  = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  model = torchvision.models.efficientnet_b2(weights=weights)
  transforms = weights.transforms()
# freezing all the layers here 
  for param in model.parameters():
      param.requires_grad = False
  model.classifier  = nn.Sequential(
          nn.Dropout(p=0.3, inplace=True) ,
          nn.Linear(in_features=1408, out_features=class_num, bias=True)
              )
  return model ,transforms

  from torchvision import transforms
from torchsummary import summary

model , transformer = transferlearning(3)

num_epoch = 10
learning_rate  = 1e-3
batch_size = 32
hidden_units = 10
data_transform = transformer

train_dir = "/content/data/pizza_steak_sushi/train"
test_dir = "/content/data/pizza_steak_sushi/test"

train_dataloader , test_dataloader , class_name = create_dataloader(train_dir , test_dir ,data_transform ,batch_size , 1 )

model =  model

#printing the summary of the model here
#summary(model , (3,224,224))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters() , lr = learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(device)
result  = Train(epochs=num_epoch ,model= model , train_dataloader=train_dataloader , test_dataloader=test_dataloader , loss_fn= loss_fn , optimizer=optimizer , device=device )
