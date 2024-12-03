#this file will contain the neural networks for the prey and predator.
#importing all the necessary modules from pytorch and the program
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi
#here we make the neural network for the prey

class prey_network(nn.Module):
  def __init__(self,n_actions):
    super().__init__()

    #makes sure that the model uses the GPU if it's available, otherwise it uses the cpu
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)
    #adding the activation functions such as relu and sigmoid
    #relu converts a negative input into 0, or just outputs the input if positive
    self.relu = nn.ReLU()
    
    #for the neural network, we will be using a fully connected neural network
    self.input_layer = nn.Linear(6,64) #6 inputs, 64 outputs
    self.hidden_layer1 = nn.Linear(64,32) #64 inputs, 32 outputs
    self.hidden_layer2 = nn.Linear(32,16) #32 inputs, 16 outputs
    self.output_layer = nn.Linear(16,n_actions)

  #the forward function of the neural network
  def forward(self,input):
    X = self.input_layer(input)
    X = self.relu(X)
    X = self.hidden_layer1(X)
    X = self.relu(X)
    X = self.hidden_layer2(X)
    #we are not going to use the activation function just yet, as they will be different for the actor and critic.
    return X
#this is what actually contains the actor and the critic, it will use the network made above.

class prey_agent(object):
  #alpha is the learning rate for the actor, beta is the learning rate for the critic, and gamma is the discount rate (how much you prioritise current rewards to future rewards)
  def __init__(self,alpha,beta,current_pos,closest_predator,closest_food,speed,gamma=0.5):
     #getting the relevant positions
    self.current_pos = current_pos
    self.closest_predator = closest_predator
    self.closest_food = closest_food
    self.speed = speed
    #splitting the x and y co-ordinates
    self.current_x,self.current_y = self.current_pos.x(),self.current_pos.y()
    self.predator_x,self.predator_y = self.closest_predator.x(),self.current_pos.y()
    self.food_x,self.food_y = self.closest_food.x(),self.closest_food.y()
    #this is what we will input into the model
    self.locations = torch.tensor([self.current_x,self.current_y,self.predator_x,self.predator_y,self.food_x,self.food_y], dtype=torch.float)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.log_probs = None
    self.softmax = nn.Softmax()

    self.actor_network = prey_network(n_actions=2)
       #intialising the optimiser
    self.actor_optim = optim.Adam(self.actor_network.parameters(),self.alpha)

    self.critic_network = prey_network(n_actions=1)
    self.critic_optim = optim.Adam(self.critic_network.parameters(),self.beta)
      


  #in order to actually choose what to do, the prey will sample the action from a normal distribution
  def choose_action(self):

    #inputs needed for the normal distribution mu is the mean, sigma is the standard deviation
    squished_output = self.softmax(self.actor_network.forward(self.locations))
    sigma,mu = squished_output[0],squished_output[1]
    #makes sure that the standard deviation is always positive
    sigma = torch.exp(sigma)
    action_probs = torch.distributions.Normal(mu,sigma) #the probablity for each action
    probs = action_probs.sample(sample_shape=torch.Size([1])) 
    self.log_probs = action_probs.log_prob(probs).to(self.actor_network.device)
    #hopefully getting the required outputs from the network here
    action = self.softmax(probs)
    moving_speed = action[0].item() * self.speed
    angle = action[1].item() *2*pi
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
    self.actor_optim.step()
    self.critic_optim.step()
    





