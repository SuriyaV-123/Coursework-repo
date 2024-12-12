#importing the relevant pyqt6 modules
from PyQt6.QtWidgets import QGraphicsEllipseItem,QGraphicsLineItem
from PyQt6.QtGui import QTransform,QPen
from PyQt6.QtCore import Qt, QPointF,QTimer,QRectF
from neural_networks import prey_agent
import torch
#this is for the temporary movement
from random import randint
#This is only used for the rays to calculate where they end
import math

#This file will store the classes for prey, predator,disease and food


class Prey(QGraphicsEllipseItem):
    def __init__(self,speed,max_energy,energy_use,attack,gen,mutation):
      #this sets the size of the prey
      super().__init__(0,0,20,20)
      #first we set the shape and colour of the prey
      self.setBrush(Qt.GlobalColor.green)
      #makes the prey on the bottom layer
      self.setZValue(0)
      #now we add the attributes that we want to.
      #how fast it moves
      self.speed = speed
      #the angle that the prey moves at
      self.angle = None
      #the maximum amount of energy it can use
      self.max_energy = max_energy
      #the energy that it uses up for each action
      self.energy_use = energy_use
      #how much damage it can deal
      self.attack = attack
      #which generation it comes from (not in terms of the neural network)
      self.gen = gen
      #its current energy
      self.energy = self.max_energy
      #the chance for the children to change its stats when born
      self.mutation_chance = mutation
      #the stats that it will pass down to children
      self.stats = [self.speed,self.max_energy,self.energy_use,self.attack]
      #its current position and co-ordinates
      self.current_pos = self.pos()
      self.x_pos = self.current_pos.x()
      self.y_pos = self.current_pos.y()
      #how much children it has had
      self.children = 0
      #whether it is currently being attacked or not
      self.being_attacked = False
      #we add a list of the rays that the prey currently has
      self.rays = []
      #the position of the closest prey,predator and food
      self.closest_predator = (None,None)
      self.closest_prey = (None,None)
      self.closest_food = (None,None)
      #this is the agent for the prey, this will allow the prey to learn and move
      self.prey_agent = prey_agent(0.5,0.5,self.speed,0.5)

   
    def __repr__(self):
       return "prey"
    #we make the energy accessible to other classes so that it can be set, e.g. when attacked by a predator
    def get_energy(self):
       return self.energy
    
    def set_energy(self,new_energy):
       self.energy = new_energy

#adding a method so that the rays can be added to the prey
    def add_rays(self):
      for i in range(7):
         #make sure that the rays are evenly spread along the circle
         angle = i * (360/8)
         #the cos and sin function only uses radians
         max_x = 100 * math.cos(math.radians(angle))
         max_y = 100 * math.sin(math.radians(angle))
         #since the circles in pyqt6 don't have a width and height method, but the bounding rectangle does,
         #we will use the bounding rectangle's height and with for the centre
         rect = self.boundingRect()
         centre_x = rect.width() / 2
         centre_y = rect.height() / 2
         #now we instantiate the ray
         ray = Ray(self,centre_x, centre_y, max_x, max_y)
         self.rays.append(ray)

    def move(self,min_x,max_x,min_y,max_y):
      min_x = min_x
      max_x = max_x
      min_y = min_y
      max_y = max_y
      if self.closest_predator == (None,None) or self.closest_food == (None,None):
         self.detect()
         self.random_move(min_x,max_x,min_y,max_y)
      
      else:

         self.prey_agent.update_locations(self.current_pos,self.closest_predator,self.closest_food)
         old_state = torch.tensor([self.x_pos,self.y_pos,self.closest_predator[0],self.closest_predator[1],self.closest_food[0],self.closest_food[1]], dtype=torch.float)
   #the moving speed and angle is chosen from the prey agent
         moving_speed, angle = prey_agent.choose_action()
         self.setRotation(angle)
         direction = QPointF(math.cos(angle),math.sin(angle))
         self.new_pos = QPointF(self.current_pos + direction * moving_speed)
         #because I don't want to mess with the learning too much, when the species reach one end of the map, they teleport to the other side
         teleport_x = False
         teleport_y = False
         if self.new_pos.x() > max_x:
            new_x = min_x + (self.new_pos.x()-max_x)
            teleport_x = True

         elif self.new_pos.x() < min_x:
            new_x = max_x - self.new_pos.x()
            teleport_x = True

         if self.new_pos.y() > max_y:
            new_y = min_y + (self.new_pos.y()-max_y)
            teleport_y = True
         
         elif self.new_pos.y() < min_y:
            new_y = max_y - self.new_pos.y()
            teleport_y = True
         
         if teleport_x is True:
            self.new_pos = QPointF(new_x,self.y_pos)

         if teleport_y is True:
            self.new_pos = QPointF(self.x_pos,new_y)
         
   #use detect
         self.detect()
         self.prey_agent.update_locations(self.current_pos,self.closest_predator,self.closest_food)      
         new_state = torch.tensor([self.new_pos.x(),self.new_pos.y(),self.closest_predator[0],self.closest_predator[1],self.closest_food[0],self.closest_food[1]], dtype=torch.float)
         self.setPos(self.new_pos)
         self.agent_learn(old_state,new_state)
         self.current_pos = self.new_pos


      #this is the loop for the agent learning, this will be called at the end of the move function
    def agent_learn(self,old_state,new_state):
       closest_predator_x = self.closest_predator[0]
       closest_predator_y = self.closest_predator[1]
       closest_food_x = self.closest_food[0]
       closest_food_y = self.closest_food[1]
       #the predator and food distance is calculated using pythagoras
       predator_distance = math.sqrt((closest_predator_x-self.x_pos)**2 + (closest_predator_y-self.y_pos)**2)
       food_distance = math.sqrt((closest_food_x-self.x_pos)**2 + (closest_food_y-self.y_pos)**2)
       #the reward is calculated as the distance between the predator and the food. This encourages the prey to try and run away from the predator and closer to the food.
       reward = predator_distance - food_distance
       prey_agent.learn(self,old_state,reward,new_state)

#now we make a temporary method so the prey can move

    def random_move(self,min_x,max_x, min_y, max_y):
       #first we move on the x-axis
       if randint(1,2) == 1:
          #this makes sure it doesn't go beyond the borders of the screen
          if (self.x_pos + self.speed) > max_x:
             self.x_pos = max_x - (self.speed + self.x_pos - max_x)
          else:
             self.x_pos += self.speed
       else:
          if (self.x_pos - self.speed) < min_x:
             self.x_pos = self.speed - self.x_pos
          else:
             self.x_pos -= self.speed
     
      #now we move on the y-axis
       if randint(1,2) == 1:
          #this makes sure it doesn't go beyond the borders of the screen
          if (self.y_pos + self.speed) > max_y:
             self.y_pos = max_y - (self.speed + self.y_pos - max_y)
          else:
             self.y_pos += self.speed
       else:
          if (self.y_pos - self.speed) < min_y:
             self.y_pos = self.speed - self.y_pos
          else:
             self.y_pos -= self.speed

       self.setPos(self.x_pos,self.y_pos)
       self.energy -= self.energy_use * 0.1
    #this happens if the prey runs out of energy, they die and are removed from the scene.
    def die(self,scene):
       if self.energy <= 0:
          dead_prey = dead_prey(self.max_energy,self.current_pos)
          scene.addItem(dead_prey)
          scene.removeItem(self)
          
    #this is the code so that a new instance of prey is made if the prey has enough energy
    def reproduce(self,scene):
       if self.energy > 50:
          #I copy the current prey's stats so its stats won't change, but they can still be changed 
          temp_stats = self.stats
          for stat in temp_stats:
             #if the number generated is below or equal to the mutation chance, then the stat will change 
             if (randint(0,100)) <= self.mutation_chance:
                if randint(1,2) == 1:
                   stat += 5
                else:
                   stat -= 5
          #now we instatiate the child here with the stats that it should have
          child = Prey(temp_stats[0],temp_stats[1],temp_stats[2],temp_stats[3],(self.gen + 1))
          scene.addItem(child)


    def eat(self, scene):
       self.detect()
       print(self.closest_food)
       if self.closest_food != (None,None):
         temp_speed = self.speed
         #this will check if there is food within a distance of 5 of the prey
         food_x = self.closest_food[0]
         food_y = self.closest_food[1]
         food_distance = math.sqrt((food_x-self.x_pos)**2 + (food_y-self.y_pos)**2)
         if food_distance <= 30:
          self.speed = 0
          pos = QPointF(food_x,food_y)
          pos_items = scene.items(QRectF(food_x-1,food_y-1,2,2))
          food = None
          print(pos)
          print(pos_items)
          for item in pos_items:
             print(type(item))
             if isinstance(item,Food):
               print('yay')
               food = item
               break
             
          if food is not None:
            extra_energy = food.get_energy()
            self.energy += extra_energy
            self.speed = temp_speed
            print('yayaya')
            scene.removeItem(food)
          else:
            print(':(')

       else:
          pass
    
       
    def defend(self,scene):
       #first it checks whether it is being attacked or not
       if self.being_attacked is True:
          #then it finds the position of the attacker within the correct range
          for x_coord in range(self.x_pos-3,self.x_pos+4):
           for y_coord in range(self.y_pos-3,self.y_pos+4):
             pos = QPointF(x_coord,y_coord)
             entity = scene.itemAt(pos, QTransform())
             if entity.__repr__() == "predator":
                #it will get the attacker's current energy
                enemy_energy = entity.get_energy()
                #it will then "deal damage" to the predator
                enemy_energy -= self.attack
                entity.set_energy()

#this takes in all the objects that the ray has detected
    def detect(self):
      #reset the lists so it doesn't affect the next iterations
      self.prey_seen = []
      self.predators_seen = []
      self.food_seen = []
      #iterate through all the rays that are connected to the prey
      for ray in self.rays:
          ray.check_collision()
          prey_detected, predators_detected, food_detected = ray.get_lists()
          #gets all the prey, predators and food seen by the rays and adds them to the lists.
          for prey in prey_detected:
             self.prey_seen.append(prey)
         
          for predator in predators_detected:
             self.predators_seen.append(predator)
          
          for food in food_detected:
             self.food_seen.append(food)
            
       #next, check what is the closest of each prey, predator and food
      if self.prey_seen != []:
         closest_prey_distance = 9999999999
         for prey in prey_detected:
            prey_pos = prey[1]
            prey_x_pos = prey_pos.x()
            prey_y_pos = prey_pos.y()
            #using pythagoras to find it
            current_distance = math.sqrt((prey_x_pos-self.x_pos)**2 + (prey_y_pos-self.y_pos)**2)
            if current_distance < closest_prey_distance:
               closest_prey_distance = current_distance
               self.closest_prey = (prey_x_pos,prey_y_pos)

      if self.predators_seen != []:
         closest_predator_distance = 9999999999
         for predator in predators_detected:
            predator_pos = predator[1]
            predator_x_pos = predator_pos.x()
            predator_y_pos = predator_pos.y()
            current_distance = math.sqrt((predator_x_pos-self.x_pos)**2 + (predator_y_pos-self.y_pos)**2)
            if current_distance < closest_predator_distance:
               closest_predator_distance = current_distance
               self.closest_predator = (predator_x_pos,predator_y_pos)
      
      if self.food_seen != []:
         closest_food_distance = 9999999999
         for food in food_detected:
            food_pos = food[1]
            food_x_pos = food_pos.x()
            food_y_pos = food_pos.y()
            current_distance = math.sqrt((food_x_pos-self.x_pos)**2 + (food_y_pos-self.y_pos)**2)
            if current_distance < closest_food_distance:
               closest_food_distance = current_distance
               self.closest_food = (food_x_pos,food_y_pos)


#This is the class for the predator, it will be very similar to the one for prey

class Predator(QGraphicsEllipseItem):
    def __init__(self,speed,max_energy,energy_use,attack,gen,mutation):
      #this sets the size of the predator
      super().__init__(0,0,40,40)
      #first we set the shape and colour of the predator
      self.setBrush(Qt.GlobalColor.red)
      self.setZValue(2)
      #now we add the attributes
      #how fast it moves
      self.speed = speed
      #the maximum amount of energy it can use
      self.max_energy = max_energy
      #the energy that it uses up for each action
      self.energy_use = energy_use
      #how much damage it can deal
      self.attack = attack
      #which generation it comes from (not in terms of the neural network)
      self.gen = gen
      #its current energy
      self.energy = self.max_energy
      #the chance for the children to change its stats when born
      self.mutation_chance = mutation
      #the stats that it will pass down to children
      self.stats = [self.speed,self.max_energy,self.energy_use,self.attack]
      #its current position and co-ordinates
      self.current_pos = self.pos()
      self.x_pos = self.current_pos.x()
      self.y_pos = self.current_pos.y()
      #how much children it has had
      self.children = 0
      #whether it is currently being attacked or not
      self.being_attacked = False
      #we add a list of the rays that the prey currently has
      self.rays = []
      #the position of the closest prey,predator and food
      self.closest_predator = (None,None)
      self.closest_prey = (None,None)
      self.closest_food = (None,None)
      #a list of all the prey in the predator's vision
      self.prey_seen = []
      #a list of all the predators in the predator's vision
      self.predators_seen = []
      #a list of all the food in the predator's vision
      self.food_seen = []

   
    def __repr__(self):
       return "predator"
    #we make the energy accessible to other classes so that it can be set, e.g. when attacked by a prey
    def get_energy(self):
       return self.energy
    
    def set_energy(self,new_energy):
       self.energy = new_energy

#adding a method so that the rays can be added to the predator
    def add_rays(self):
      for i in range(20):
         #make sure that the rays are in an arc
         angle = i * (360/180)
         #the cos function only uses radians
         max_x = 300 * math.cos(math.radians(angle))
         max_y = 300 * math.sin(math.radians(angle))
         #since the circles in pyqt6 don't have a width and height method, but the bounding rectangle does,
         #we will use the bounding rectangle's height and with for the centre
         rect = self.boundingRect()
         centre_x = rect.width() / 2
         centre_y = rect.height() / 2
         #now we instantiate the ray
         ray = Ray(self,centre_x, centre_y, max_x, max_y)
         #add it to the list of rays
         self.rays.append(ray)

#now we make a temporary method so the predator can move
    
    def move(self,min_x,max_x, min_y, max_y):
       #first we move on the x-axis
       if randint(1,2) == 1:
          #this makes sure it doesn't go beyond the borders of the screen
          if (self.x_pos + self.speed) > max_x:
             self.x_pos = max_x - (self.speed + self.x_pos - max_x)
          else:
             self.x_pos += self.speed
       else:
          if (self.x_pos - self.speed) < min_x:
             self.x_pos = self.speed - self.x_pos
          else:
             self.x_pos -= self.speed
     
      #now we move on the y-axis
       if randint(1,2) == 1:
          #this makes sure it doesn't go beyond the borders of the screen
          if (self.y_pos + self.speed) > max_y:
             self.y_pos = max_y - (self.speed + self.y_pos - max_y)
          else:
             self.y_pos += self.speed
       else:
          if (self.y_pos - self.speed) < min_y:
             self.y_pos = self.speed - self.y_pos
          else:
             self.y_pos -= self.speed

       self.setPos(self.x_pos,self.y_pos)
       self.energy -= self.energy_use * 0.1
    #this happens if the predator runs out of energy, they die and are removed from the scene.
    def die(self,scene):
       if self.energy <= 0:
          scene.removeItem(self)
          
    #this is the code so that a new instance of predator is made if the predator has enough energy
    def reproduce(self,scene):
       if self.energy > 50:
          #I copy the current predator's stats so its stats won't change, but they can still be altered for the child's stats
          temp_stats = self.stats
          for stat in temp_stats:
             #if the number generated is below or equal to the mutation chance, then the stat will change 
             if (randint(0,100)) <= self.mutation_chance:
                if randint(1,2) == 1:
                   stat += 5
                else:
                   stat -= 5
          #now we instatiate the child here with the stats that it should have
          child = Predator(temp_stats[0],temp_stats[1],temp_stats[2],temp_stats[3],(self.gen + 1))
          scene.addItem(child)


    def eat(self, scene):
       #because we want to stop the prey from moving while eating, we temporarily reduce it's speed to 0 and set it back to the original speed afterwards
       temp_speed = self.speed
       #this will check if there is food within 3 coordinates of the prey
       for x_coord in range(self.x_pos-3,self.x_pos+4):
          for y_coord in range(self.y_pos-3,self.y_pos+4):
             pos = QPointF(x_coord,y_coord)
       #this checks if there is food within reach and will take the energy from it
             entity = scene.itemAt(pos, QTransform())
             if entity.__repr__() == "food":
              self.speed = 0
              extra_energy = entity.get_energy()
              self.energy += extra_energy
       self.speed = temp_speed
    
    def defend(self,scene):
       #first it checks whether it is being attacked or not
       if self.being_attacked is True:
          #then it finds the position of the attacker within the correct range
          for x_coord in range(self.x_pos-3,self.x_pos+4):
           for y_coord in range(self.y_pos-3,self.y_pos+4):
             pos = QPointF(x_coord,y_coord)
             entity = scene.itemAt(pos, QTransform())
             if entity.__repr__() == "prey":
                #it will get the attacker's current energy
                enemy_energy = entity.get_energy()
                #it will then "deal damage" to the predator
                enemy_energy -= self.attack
                entity.set_energy()

#this takes in all the objects that the ray has detected
    def detect(self):
       self.prey_seen = []
       self.predators_seen = []
       self.food_seen = []
       #iterate through all the rays that are connected to the prey
       for ray in self.rays:
          ray.check_collision()
          prey_detected, predators_detected, food_detected = ray.get_lists()
          #gets all the prey, predators and food seen by the rays and adds them to the lists.
          for prey in prey_detected:
             self.prey_seen.append(prey)
         
          for predator in predators_detected:
             self.predators_seen.append(predator)
          
          for food in food_detected:
             self.food_seen.append(food)
            
       #next, check what is the closest of each prey, predator and food
       if self.prey_seen != []:
         closest_prey_distance = 9999999999
         for prey in self.prey_seen:
          prey_pos = prey[1]
          prey_x_pos = prey_pos.x()
          prey_y_pos = prey_pos.y()

          current_distance = math.sqrt((prey_x_pos-self.x_pos)**2 + (prey_y_pos-self.y_pos)**2)
          if current_distance < closest_prey_distance:
             closest_prey_distance = current_distance
             self.closest_prey = (prey_x_pos,prey_y_pos)

       if self.predators_seen != []:     
         closest_predator_distance = 9999999999
         for predator in self.predators_seen:
            predator_pos = predator[1]
            predator_x_pos = predator_pos.x()
            predator_y_pos = predator_pos.y()
            current_distance = math.sqrt((predator_x_pos-self.x_pos)**2 + (predator_y_pos-self.y_pos)**2)
            if current_distance < closest_predator_distance:
               closest_predator_distance = current_distance
               self.closest_predator = (predator_x_pos,predator_y_pos)
       
       if self.food_seen != []:
         closest_food_distance = 9999999999
         for food in self.food_seen:
            food_pos = food[1]
            food_x_pos = food_pos.x()
            food_y_pos = food_pos.y()
            current_distance = math.sqrt((food_x_pos-self.x_pos)**2 + (food_y_pos-self.y_pos)**2)
            if current_distance < closest_food_distance:
               closest_food_distance = current_distance
               self.closest_food = (food_x_pos,food_y_pos)
       print(self.closest_prey)

#Class for when prey die, so that predators can eat them

class dead_prey(QGraphicsEllipseItem):
   def __init__(self,energy_stored,pos):
      super().__init__(0,0,10,10)
      self.setZValue(3)
      #set the colour to grey, to show that the prey has died.
      self.setBrush(Qt.GlobalColor.gray)
      self.energy_stored = energy_stored
      self.coord = pos
      




#in order for detection to work for the prey and predators, I will be making a new class for 'rays'
#whenever an object collides with a ray, it will report the position of the object to the creature it's attached to
class Ray(QGraphicsLineItem):
   def __init__(self,attached_prey,centre_x,centre_y,max_x,max_y):
      super().__init__(centre_x,centre_y,max_x,max_y,attached_prey)
      pen = QPen(Qt.GlobalColor.black)
      pen.setWidth(3)
      self.setPen(pen)
      self.detected_prey = []
      self.detected_predators = []
      self.detected_food = []
      self.setZValue(1)

   #returns the objects that the ray has
   def get_lists(self):
      return self.detected_prey, self.detected_predators, self.detected_food
   

   def check_collision(self):
      self.detected_items = self.collidingItems()
      #resetting the viewed objects so it doesn't think there's any objects in sight when there aren't.
      self.detected_prey,self.detected_predators,self.detected_food = [],[],[]
      for item in self.detected_items:
         if isinstance(item,Predator):
            self.detected_predators.append([item,item.pos()])
         elif isinstance(item,Food):
            self.detected_food.append([item,item.pos()])
         elif isinstance(item,Prey):
            self.detected_prey.append([item,item.pos()])

#making the class for food
class Food(QGraphicsEllipseItem):
   def __init__(self):
      super().__init__(0,0,10,10)
      self.setBrush(Qt.GlobalColor.yellow)
      self.setZValue(400)
      #for now, giving any necessary attributes arbitrary values
      self.energy_stored = 30
      self.current_pos = self.pos()
      self.x_pos = self.current_pos.x()
      self.y_pos = self.current_pos.y()

   def __repr__(self):
      return "food"
   
   def get_energy(self):
      return self.energy_stored