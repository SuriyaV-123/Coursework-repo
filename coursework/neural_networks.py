#this file will contain the neural networks for the prey and predator.
#importing all the necessary modules from pytorch and the program
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi
from PyQt6.QtCore import QObject, pyqtSignal
#here we make the neural network for the prey

class prey_network(nn.Module):
  def __init__(self,n_actions):
    super().__init__()


    #adding the activation functions such as relu and sigmoid
    #relu converts a negative input into 0, or just outputs the input if positive
    self.relu = nn.ReLU()
    
    #for the neural network, we will be using a fully connected neural network
    self.input_layer = nn.Linear(6,64) #6 inputs, 64 outputs
    self.hidden_layer1 = nn.Linear(64,32) #64 inputs, 32 outputs
    self.hidden_layer2 = nn.Linear(32,16) #32 inputs, 16 outputs
    self.output_layer = nn.Linear(16,n_actions)

    #makes sure that the model uses the GPU if it's available, otherwise it uses the cpu
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

# Initialize weights
    for layer in [self.input_layer, self.hidden_layer1, self.hidden_layer2, self.output_layer]:
      nn.init.xavier_uniform_(layer.weight)
      nn.init.zeros_(layer.bias)  
      #also zero the bias to start with.  

  #the forward function of the neural network
  def forward(self,input):
    X = self.input_layer(input)
    X = self.relu(X)
    X = self.hidden_layer1(X)
    X = self.relu(X)
    X = self.hidden_layer2(X)
    output = self.output_layer(X)
    #we are not going to use the activation function just yet, as they will be different for the actor and critic.
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

    self.critic_network = prey_network(n_actions=1)
    self.critic_optim = optim.Adam(self.critic_network.parameters(),self.beta)
  #I added this function to make sure that the value of the closest predator and food wouldn't be null, so it could be converted into a QPointF
  def update_locations(self,current_pos,closest_predator,closest_food):
    self.current_pos_x = current_pos.x()
    self.current_pos_y = current_pos.y()
    self.closest_predator_x = closest_predator[0]
    self.closest_predator_y = closest_predator[1]
    self.closest_food_x = closest_food[0]
    self.closest_food_y = closest_food[1]
    self.locations = torch.tensor([self.current_pos_x,self.current_pos_y,self.closest_predator_x,self.closest_predator_y,self.closest_food_x,self.closest_food_y],dtype=torch.float,device=self.actor_network.device)

  #in order to actually choose what to do, the prey will sample the action from a normal distribution
  def choose_action(self):
    
    #inputs needed for the normal distribution mu is the mean, sigma is the standard deviation
    #the dim is set to -1 to make sure that whatever is tensor size is passed through, the softmax function will still work.
    output = self.actor_network.forward(self.locations)
    mu,sigma = output[0],output[1]
    mu = torch.tanh(mu)

    #makes sure that the standard deviation is always positive and that the value does not get out of hand
    sigma = torch.clamp(torch.exp(sigma),min=1e-2,max=10)
    action_probs = torch.distributions.Normal(mu,sigma) #the probablity for each action
    probs = action_probs.sample(sample_shape=torch.Size([2]))
    self.log_probs = action_probs.log_prob(probs).sum().to(self.actor_network.device)

    #hopefully getting the required outputs from the network here
    moving_speed = probs[0].item() * self.speed
    angle = probs[1].item() *2*pi
    #should emit the finish signal now that it's done.
    #self.finished.emit()
    return moving_speed,angle
    
    
#now we update the weights and biases within the networks
  def learn(self,first_state,reward,second_state):
    #resets the gradients so there is no unneccesary info when training the networks
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()

    new_critic_value = self.critic_network.forward(second_state)
    critic_value = self.critic_network.forward(first_state)
    reward = torch.tensor(reward, dtype=torch.float).to(self.actor_network.device)
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




 # def count_parameters(self):
 #   return sum(p.numel() for p in self.actor_network.parameters() if p.requires_grad)