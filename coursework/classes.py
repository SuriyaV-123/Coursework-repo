#importing the relevant pyqt6 modules
from PyQt6.QtWidgets import QGraphicsEllipseItem,QGraphicsLineItem
from PyQt6.QtGui import QTransform,QPen
from PyQt6.QtCore import Qt, QPointF
#this is for the temporary movement
from random import randint
#This is only used for the rays to calculate where they end
import math
#used for methods which happen after a certain time in the simulation.
from threading import Timer
#This file will store the classes for prey, predator and disease


class Prey(QGraphicsEllipseItem):
    def __init__(self,speed,max_energy,energy_use,attack,gen,mutation):
      #this sets the size of the prey
      super().__init__(0,0,20,20)
      #first we set the shape and colour of the prey
      self.setBrush(Qt.GlobalColor.green)
      #now we add the attributes that we want to. For now I will add arbitrary values
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
      #a list of all the prey in the prey's vision
      self.prey_seen = []
      #a list of all the predators in the prey's vision
      self.predators_seen = []
      #a list of all the food in the prey's vision
      self.food_seen = []

   
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
         #the cos function only uses radians
         max_x = 100 * math.cos(math.radians(angle))
         max_y = 100 * math.sin(math.radians(angle))
         #since the circles in pyqt6 don't have a width and height method, but the bounding rectangle does,
         #we will use the bounding rectangle's height and with for the centre
         rect = self.boundingRect()
         centre_x = rect.width() / 2
         centre_y = rect.height() / 2
         #now we instantiate the ray
         ray = Ray(self,centre_x, centre_y, max_x, max_y)
         #add it to the list of rays
         self.rays.append(ray)




      
#now we make a temporary method so the prey can move

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
      closest_prey_distance = 9999999999
      for prey in prey_detected:
          prey_pos = prey[1]
          prey_x_pos = prey_pos.x()
          prey_y_pos = prey_pos.y()
          #using pythagorus to find it
          current_distance = math.sqrt((prey_x_pos-self.x_pos)**2 + (prey_y_pos-self.y_pos)**2)
          if current_distance < closest_prey_distance:
             closest_prey_distance = current_distance
             self.closest_prey = (prey_x_pos,prey_y_pos)
             
      closest_predator_distance = 9999999999
      for predator in predators_detected:
          predator_pos = predator[1]
          predator_x_pos = predator_pos.x()
          predator_y_pos = predator_pos.y()
          current_distance = math.sqrt((predator_x_pos-self.x_pos)**2 + (predator_y_pos-self.y_pos)**2)
          if current_distance < closest_predator_distance:
             closest_predator_distance = current_distance
             self.closest_predator = (predator_x_pos,predator_y_pos)
       
      closest_food_distance = 9999999999
      for food in food_detected:
          food_pos = food[1]
          food_x_pos = food_pos.x()
          food_y_pos = food_pos.y()
          current_distance = math.sqrt((food_x_pos-self.x_pos)**2 + (food_y_pos-self.y_pos)**2)
          if current_distance < closest_food_distance:
             closest_food_distance = current_distance
             self.closest_food = (food_x_pos,food_y_pos)

      print(self.closest_predator)


#This is the class for the predator, it will be very similar to the one for prey

class Predator(QGraphicsEllipseItem):
    def __init__(self,speed,max_energy,energy_use,attack,gen,mutation):
      #this sets the size of the predator
      super().__init__(0,0,40,40)
      #first we set the shape and colour of the predator
      self.setBrush(Qt.GlobalColor.red)
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
       closest_prey_distance = 9999999999
       for prey in self.prey_seen:
          prey_pos = prey[1]
          prey_x_pos = prey_pos.x()
          prey_y_pos = prey_pos.y()

          current_distance = math.sqrt((prey_x_pos-self.x_pos)**2 + (prey_y_pos-self.y_pos)**2)
          if current_distance < closest_prey_distance:
             closest_prey_distance = current_distance
             self.closest_prey = (prey_x_pos,prey_y_pos)
             
       closest_predator_distance = 9999999999
       for predator in self.predators_seen:
          predator_pos = predator[1]
          predator_x_pos = predator_pos.x()
          predator_y_pos = predator_pos.y()
          current_distance = math.sqrt((predator_x_pos-self.x_pos)**2 + (predator_y_pos-self.y_pos)**2)
          if current_distance < closest_predator_distance:
             closest_predator_distance = current_distance
             self.closest_predator = (predator_x_pos,predator_y_pos)
       
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

   #returns the objects that the ray has
   def get_lists(self):
      return self.detected_prey, self.detected_predators, self.detected_food
   

   def check_collision(self):
      self.detected_items = self.collidingItems()
      #resetting the viewed objects so it doesn't think there's any objects in sight when there aren't.
      self.detected_prey,self.detected_predators,self.detected_food = [],[],[]
      for item in self.detected_items:
         if item.__repr__() == "predator":
            self.detected_predators.append([item,item.pos()])
         elif item.__repr__() == "food":
            self.detected_food.append([item,item.pos()])
         elif item.__repr__() == "prey":
            self.detected_prey.append([item,item.pos()])

#making the class for food
class Food(QGraphicsEllipseItem):
   def __init__(self):
      super().__init__(0,0,10,10)
      self.setBrush(Qt.GlobalColor.yellow)
      #for now, giving any necessary attributes arbitrary values
      self.energy_stored = 30
      self.current_pos = self.pos()
      self.x_pos = self.current_pos.x()
      self.y_pos = self.current_pos.y()



      