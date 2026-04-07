#create dataset
features = torch.rand(10,5)

#create model
model = Model(features.shape[1])   #apne model ka object banana hai aur simply as a function call kardena hai

#call model for forward pass
#model.forward(features)
model(features)   #automatically triggers forward method via magic method inside nn Module
