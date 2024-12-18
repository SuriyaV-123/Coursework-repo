#this file will contain the neural networks for the prey and predator.
#importing all the necessary modules from pytorch and the program
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi
from PyQt6.QtCore import QPointF
#here we make the neural network for the prey

class prey_network(nn.Module):
  def __init__(self,n_actions):
    super().__init__()


    #adding the activation functions such as relu and sigmoid
    self.relu = nn.ReLU()
    
    #for the neural network, we will be using a fully connected neural network
    self.input_layer = nn.Linear(6,64) #6 inputs, 64 outputs
    self.hidden_layer1 = nn.Linear(64,32) #64 inputs, 32 outputs
    self.hidden_layer2 = nn.Linear(32,16) #32 inputs, 16 outputs
    self.output_layer = nn.Linear(16,n_actions)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

# Initialize weights
    for layer in [self.input_layer, self.hidden_layer1, self.hidden_layer2, self.output_layer]:
      #am using kaiming to distribute weights instead of xavier, as kaiming is better suited to continuous distributions
      nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
      nn.init.zeros_(layer.bias)  
      #also zero the bias to start with.  

  def forward(self,input):
    print("Input tensor details:")
    print(f"Input shape: {input.shape}")
    print(f"Input dtype: {input.dtype}")
    print(f"Input device: {input.device}")
    
    # Check for NaN values (debug stuff)
    if torch.isnan(input).any():
        print("NaN detected in input")
        print("NaN locations:")
        print(torch.isnan(input))
    
    # Check for extreme values (again debug stuff)
    print(f"Input min: {input.min()}")
    print(f"Input max: {input.max()}")

    X = self.input_layer(input)
    X = self.relu(X)
    X = self.hidden_layer1(X)
    X = self.relu(X)
    X = self.hidden_layer2(X)
    output = self.output_layer(X)
    #we are not going to use the activation function just yet, as they will be different for the actor and critic.
    #this makes sure that if the output is nan, it is converted to 0
    output = torch.nan_to_num(output, nan=0.0)
    return output
#this is what actually contains the actor and the critic, it will use the network made above.
#as the program does not respond when I try to use the neural network, i'm going to try and use threading
#the prey_agent will also act as the 'worker' and will send the signals over or something.

class prey_agent(object):
  #alpha is the learning rate for the actor, beta is the learning rate for the critic, and gamma is the discount rate (how much you prioritise current rewards to future rewards)
  def __init__(self,alpha,beta,speed,gamma=0.5):
    self.speed = speed
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.log_probs = None
    self.softmax = nn.Softmax(dim=-1)

    self.actor_network = prey_network(n_actions=2)
       #intialising the optimiser
    self.actor_optim = optim.Adam(self.actor_network.parameters(),self.alpha)
    self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optim, step_size=100, gamma=0.9)

    self.critic_network = prey_network(n_actions=1)
    self.critic_optim = optim.Adam(self.critic_network.parameters(),self.beta)
    self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optim, step_size=100, gamma=0.9)    

  def update_locations(self,current_pos,closest_predator,closest_food):
#all of this is to make sure that there is something in the closest predator and closest food, so that when it is trying to choose an action, the program doesn't crash.
    if isinstance(current_pos,QPointF) is True:
      current_pos_x = current_pos.x()
      current_pos_y = current_pos.y()
    else:
      current_pos_x = current_pos
      current_pos_y = current_pos

    if closest_predator != (None,None):

      closest_predator_x = closest_predator[0]
      closest_predator_y = closest_predator[1]
    
    else:
      closest_predator_x = 0.0
      closest_predator_y = 0.0
    
    if closest_food != (None,None):

      closest_food_x = closest_food[0]
      closest_food_y = closest_food[1]
    
    else:
      closest_food_x = 0.0
      closest_food_y = 0.0

    max_x = 600
    max_y = 900

    current_pos_x = max(min(current_pos_x / max_x, 1), -1)
    current_pos_y = max(min(current_pos_y / max_y, 1), -1)
    closest_predator_x = max(min(closest_predator_x / max_x, 1), -1)
    closest_predator_y = max(min(closest_predator_y / max_y, 1), -1)
    closest_food_x = max(min(closest_food_x / max_x, 1), -1)
    closest_food_y = max(min(closest_food_y / max_y, 1), -1)

    
    self.locations = torch.tensor([current_pos_x,current_pos_y,closest_predator_x,closest_predator_y,closest_food_x,closest_food_y],dtype=torch.float,device=self.actor_network.device)

  #in order to actually choose what to do, the prey will sample the action from a normal distribution
  def choose_action(self):
#if at any point, there is a value which has nan in it, set a default value so that the program will not crash.
    if torch.isnan(self.locations).any():
      print('An item has a value of NaN')
      self.locations = torch.zeros(6,dtype=torch.float,device=self.actor_network.device)


    #inputs needed for the normal distribution mu is the mean, sigma is the standard deviation
    
    output = self.actor_network.forward(self.locations)
    #doing the same validation for NaN for the locations onto the output
    if torch.isnan(output).any() or torch.isinf(output).any():
      output = torch.zeros(2,dtype=torch.float,device=self.actor_network.device)
      print('output has nan')

    mu,sigma = output[0],output[1]

    print(self.locations)
    print(output)
    print(f'Raw mu:{mu}')
    print(f'Raw sigma:{sigma}')
    mu = torch.tanh(mu)

    #makes sure that the standard deviation is always positive and that the value does not get out of hand
    sigma = torch.nn.functional.softplus(sigma) + 1e-2 
    print(f'Output mu:{mu}')
    print(f'Output sigma: {sigma}')
    action_probs = torch.distributions.Normal(mu,sigma) #the probablity for each action
    probs = action_probs.sample(sample_shape=torch.Size([2]))
    self.log_probs = action_probs.log_prob(probs).sum().to(self.actor_network.device)

    #hopefully getting the required outputs from the network here
    moving_speed = probs[0].item() * self.speed
    angle = probs[1].item() *360
    return moving_speed,angle
  
    
    
#the learning method for the neural network
  def learn(self,first_state,reward,second_state):
    
    #resets the gradients so there is no unneccesary info when training the networks
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()
    

    new_critic_value = self.critic_network.forward(second_state)
    critic_value = self.critic_network.forward(first_state)

    reward = torch.tensor(reward, dtype=torch.float).to(self.actor_network.device)
    reward = torch.clamp(reward, min=-50, max = 50)
    print(f"reward: {reward}")
    #delta is the temporal difference loss (a.k.a the difference between what happens and what we want to happen)
    delta = reward + self.gamma*new_critic_value - critic_value
    actor_loss = -self.log_probs * delta
    critic_loss = delta ** 2 #making sure that delta is positive
    (actor_loss + critic_loss).backward() #using backpropagation to update weights and biases
    #this should make sure that there is no exploding gradients, and keep them under control
    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.0)
    self.actor_optim.step()
    self.critic_optim.step()

    self.actor_scheduler.step()
    self.critic_scheduler.step()



class predator_network(nn.Module):
  def __init__(self,n_actions):
    super().__init__()


    #adding the activation functions such as relu and sigmoid
    self.relu = nn.ReLU()
    
    #for the neural network, we will be using a fully connected neural network
    self.input_layer = nn.Linear(6,64) #6 inputs, 64 outputs
    self.hidden_layer1 = nn.Linear(64,32) #64 inputs, 32 outputs
    self.hidden_layer2 = nn.Linear(32,16) #32 inputs, 16 outputs
    self.output_layer = nn.Linear(16,n_actions)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

# Initialize weights
    for layer in [self.input_layer, self.hidden_layer1, self.hidden_layer2, self.output_layer]:
      #am using kaiming to distribute weights instead of xavier, as kaiming is better suited to continuous distributions
      nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
      nn.init.zeros_(layer.bias)  
      #also zero the bias to start with.  

  def forward(self,input):
    print("Input tensor details:")
    print(f"Input shape: {input.shape}")
    print(f"Input dtype: {input.dtype}")
    print(f"Input device: {input.device}")
    
    # Check for NaN values (debug stuff)
    if torch.isnan(input).any():
        print("NaN detected in input")
        print("NaN locations:")
        print(torch.isnan(input))
    
    # Check for extreme values (again debug stuff)
    print(f"Input min: {input.min()}")
    print(f"Input max: {input.max()}")

    X = self.input_layer(input)
    X = self.relu(X)
    X = self.hidden_layer1(X)
    X = self.relu(X)
    X = self.hidden_layer2(X)
    output = self.output_layer(X)
    #we are not going to use the activation function just yet, as they will be different for the actor and critic.
    #this makes sure that if the output is nan, it is converted to 0
    output = torch.nan_to_num(output, nan=0.0)
    return output
#this is what actually contains the actor and the critic, it will use the network made above.
#as the program does not respond when I try to use the neural network, i'm going to try and use threading
#the prey_agent will also act as the 'worker' and will send the signals over or something.

class predator_agent(object):
  #alpha is the learning rate for the actor, beta is the learning rate for the critic, and gamma is the discount rate (how much you prioritise current rewards to future rewards)
  def __init__(self,alpha,beta,speed,gamma=0.5):
    self.speed = speed
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.log_probs = None
    self.softmax = nn.Softmax(dim=-1)

    self.actor_network = predator_network(n_actions=2)
       #intialising the optimiser
    self.actor_optim = optim.Adam(self.actor_network.parameters(),self.alpha)
    self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optim, step_size=100, gamma=0.9)

    self.critic_network = predator_network(n_actions=1)
    self.critic_optim = optim.Adam(self.critic_network.parameters(),self.beta)
    self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optim, step_size=100, gamma=0.9)    

  def update_locations(self,current_pos,closest_prey,closest_dead_prey):
#all of this is to make sure that there is something in the closest predator and closest food, so that when it is trying to choose an action, the program doesn't crash.
    if isinstance(current_pos,QPointF) is True:
      current_pos_x = current_pos.x()
      current_pos_y = current_pos.y()
    else:
      current_pos_x = current_pos
      current_pos_y = current_pos

    if closest_prey != (None,None):

      closest_prey_x = closest_prey[0]
      closest_prey_y = closest_prey[1]
    
    else:
      closest_prey_x = 0.0
      closest_prey_y = 0.0
    
    if closest_dead_prey != (None,None):

      closest_dead_prey_x = closest_dead_prey[0]
      closest_dead_prey_y = closest_dead_prey[1]
    
    else:
      closest_dead_prey_x = 0.0
      closest_dead_prey_y = 0.0

    max_x = 600
    max_y = 900

    current_pos_x = max(min(current_pos_x / max_x, 1), -1)
    current_pos_y = max(min(current_pos_y / max_y, 1), -1)
    closest_prey_x = max(min(closest_prey_x / max_x, 1), -1)
    closest_prey_y = max(min(closest_prey_y / max_y, 1), -1)
    closest_dead_prey_x = max(min(closest_dead_prey_x / max_x, 1), -1)
    closest_dead_prey_y = max(min(closest_dead_prey_y / max_y, 1), -1)

    
    self.locations = torch.tensor([current_pos_x,current_pos_y,closest_prey_x,closest_prey_y,closest_dead_prey_x,closest_dead_prey_y],dtype=torch.float,device=self.actor_network.device)

  #in order to actually choose what to do, the prey will sample the action from a normal distribution
  def choose_action(self):
#if at any point, there is a value which has nan in it, set a default value so that the program will not crash.
    if torch.isnan(self.locations).any():
      print('An item has a value of NaN')
      self.locations = torch.zeros(6,dtype=torch.float,device=self.actor_network.device)


    #inputs needed for the normal distribution mu is the mean, sigma is the standard deviation
    
    output = self.actor_network.forward(self.locations)
    #doing the same validation for NaN for the locations onto the output
    if torch.isnan(output).any() or torch.isinf(output).any():
      output = torch.zeros(2,dtype=torch.float,device=self.actor_network.device)
      print('output has nan')

    mu,sigma = output[0],output[1]

    print(self.locations)
    print(output)
    print(f'Raw mu:{mu}')
    print(f'Raw sigma:{sigma}')
    mu = torch.tanh(mu)

    #makes sure that the standard deviation is always positive and that the value does not get out of hand
    sigma = torch.nn.functional.softplus(sigma) + 1e-2 
    print(f'Output mu:{mu}')
    print(f'Output sigma: {sigma}')
    action_probs = torch.distributions.Normal(mu,sigma) #the probablity for each action
    probs = action_probs.sample(sample_shape=torch.Size([2]))
    self.log_probs = action_probs.log_prob(probs).sum().to(self.actor_network.device)

    #hopefully getting the required outputs from the network here
    moving_speed = probs[0].item() * self.speed
    angle = probs[1].item() *360
    return moving_speed,angle
  
    
    
#the learning method for the neural network
  def learn(self,first_state,reward,second_state):
    
    #resets the gradients so there is no unneccesary info when training the networks
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()
    

    new_critic_value = self.critic_network.forward(second_state)
    critic_value = self.critic_network.forward(first_state)

    reward = torch.tensor(reward, dtype=torch.float).to(self.actor_network.device)
    reward = torch.clamp(reward, min=-50, max = 50)
    #delta is the temporal difference loss (a.k.a the difference between what happens and what we want to happen)
    delta = reward + self.gamma*new_critic_value - critic_value
    actor_loss = -self.log_probs * delta
    critic_loss = delta ** 2 #making sure that delta is positive
    (actor_loss + critic_loss).backward() #using backpropagation to update weights and biases
    #this should make sure that there is no exploding gradients, and keep them under control
    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.0)
    self.actor_optim.step()
    self.critic_optim.step()

    self.actor_scheduler.step()
    self.critic_scheduler.step()

