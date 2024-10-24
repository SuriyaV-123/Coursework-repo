from PyQt6.QtWidgets import QGraphicsEllipseItem,QGraphicsLineItem
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import Qt, QPointF
from random import randint
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
      #the position of the closest predator and food
      self.closest_predator = (None,None)
      self.closest_prey = (None,None)
      #a list of all the predators in the prey's vision
      self.predators_seen = []
      #a list of all the food in the predator's vision
      self.food_seen = []

   
    def __repr__(self):
       return "prey"
    #we make the energy accessible to other classes so that it can be set, e.g. when attacked by a predator
    def get_energy(self):
       return self.energy
    
    def set_energy(self,new_energy):
       self.energy = new_energy
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

    def detect(self,scene):
       pass

#in order for detection to work for the prey and predators, I will be making a new class for 'rays'
#whenever an object collides with a ray, it will report the position of the object to the creature it's attached to
class ray(QGraphicsLineItem):
   def __init__(self,prey_in_sight,predators_in_sight,food_in_sight):
      super().__init__(0,0,50,50)
      self.setBrush(Qt.GlobalColor.black)
      self.detected_predators = []
      self.detected_food = []


   
   def check_collision(self):
      self.detected_items = self.collidingItems()
      for item in self.detected_items:
         if item.__repr__() == "predator":
            self.detected_predators.append([item,item.pos()])
         elif item.__repr__() == "food":
            self.detected_food.append([item,item.pos()])
      






