#first we import the necessary modules, mainly from pyqt6. We also import the classes from the classes file
from PyQt6.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QSlider, QHBoxLayout,QGroupBox, QCheckBox, QGraphicsScene, QGraphicsView
from PyQt6.QtCore import Qt, QTimer,QPointF
from PyQt6.QtGui import QFont,QKeyEvent
from random import randint
from classes import Prey,Predator,Food,Dead_prey
import pyqtgraph as pg
#This file will contain the classes need for the gui for the simulation, they will be used for the different windows
#I will be using the PyQt6 library to make the gui.


class Start_Window(QMainWindow):
    #This is the class for the start window, where it will contain some information about the simulation and a button to go to the next page
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prey vs predator simulation")
        layout = QVBoxLayout()
        #This contains the intro text
        text = '''
Welcome this prey vs predator simulation, where it will simulate an interaction between two species: a prey population and a predator population.
The simulation will display a graph in the top right corner displaying the populations. Before the simulation starts, you can change the settings,
to customise it to how you want it. The speed is how fast it moves, the average energy use is the energy use multiplier per action,
the max energy is the most energy it can store. The population is how many of them are there to start with, the mutation is the chance of the child's
stats changing compared to it's parent, and the attack is how much damage it can deal. The infectivity is the chance to infect a creature, and the
lethality is the amount of energy the disease takes from an infected creature. The simulation will end once one of the populations die or when you decide to end it.
You can press the spacebar to pause or unpause the simulation at anytime. '''
        intro = QLabel(text)
        #this is the button to go to the next page
        settings_button = QPushButton("Go to settings")
        #this will set the button to a certain size
        settings_button.setFixedSize(200,100)
        #this will go to the settings page once clicked
        settings_button.clicked.connect(self.show_settings)


        font = intro.font()
        #setting the font size here
        font_button = QFont('Arial',15)
        font.setPointSize(15)
        intro.setFont(font)
        settings_button.setFont(font_button)
       
        #making sure that the text is aligned in the centre of the screen
        intro.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
       
        #this makes the text and button be aligned vertically
        layout.addWidget(intro)
        layout.addWidget(settings_button)
        layout.setAlignment(settings_button,Qt.AlignmentFlag.AlignCenter)


        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)



    def show_settings(self):
         #we will create an instance of the settings window in here so we can show the window using the main menu
        self.setting_window = settings_window()
        if self.setting_window.isVisible():
          self.setting_window.hide()
        else:
          self.setting_window.show()

#now we will create the window for the settings window


class settings_window(QMainWindow):
    def __init__(self):
       super().__init__()
       self.setWindowTitle("Settings")


       #as we are going to have three boxes arranged horizontally, we need to have multiple layouts
       prey_layout = QVBoxLayout()
       predator_layout = QVBoxLayout()
       disease_layout = QVBoxLayout()
       box_layout = QHBoxLayout()
       main_layout = QVBoxLayout()
       #this is what will be displayed at the top
       title = QLabel("Settings")
       title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
       #this is the button which will start the simulation
       start_button = QPushButton("Start")
       start_button.setFixedSize(100,100)
       button_font = QFont('Arial', 15)
       start_button.setFont(button_font)
       start_button.clicked.connect(self.update_value)
       start_button.clicked.connect(self.start_sim)


       #changing the font size for the title
       title_font = title.font()
       title_font.setPointSize(15)
       title.setFont(title_font)
       #making the separate boxes for the different species and changing the size of the boxes.
       prey_box = QGroupBox()
       prey_box.setFixedSize(400,400)
       predator_box = QGroupBox()
       predator_box.setFixedSize(400,400)
       disease_box = QGroupBox()
       disease_box.setFixedSize(400,400)
       #giving each box a title
       prey_title = QLabel("Prey")
       predator_title = QLabel("Predator")
       disease_title = QLabel("Disease")
       prey_layout.addWidget(prey_title)
       predator_layout.addWidget(predator_title)
       disease_layout.addWidget(disease_title)
       #these are the sliders and checkboxes used in the boxes
       #by using sliders and checkboxes, it removes the possiblity of any syntax errors caused by user inputs, as the program defines the limits and input type.
       #originally, I was going to have one set of sliders which would be used in both boxes, but that didn't work so now I have to make them separately
       self.prey_speed_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.prey_population_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.prey_energy_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.prey_mutation_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.prey_max_energy_slider = QSlider(Qt.Orientation.Horizontal)
       self.prey_attack_slider = QSlider(Qt.Orientation.Horizontal)

       self.predator_speed_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.predator_population_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.predator_energy_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.predator_mutation_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.predator_max_energy_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.predator_attack_slider = QSlider(Qt.Orientation.Horizontal)

       self.infectivity_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.lethality_slider = QSlider(Qt.Orientation.Horizontal, self)
       self.prey_affected = QCheckBox()
       self.predator_affected = QCheckBox()
       self.prey_affected.setCheckState(Qt.CheckState.Checked)
       self.predator_affected.setCheckState(Qt.CheckState.Checked)
       self.prey_affected.setText("Prey")
       self.predator_affected.setText("Predator")

       #creating a set of labels so the user can see what value they are choosing
       prey_speed_label = QLabel()
       prey_population_label = QLabel()
       prey_energy_label = QLabel()
       prey_max_label = QLabel()
       prey_mutation_label = QLabel()
       prey_attack_label = QLabel()

       predator_speed_label = QLabel()
       predator_population_label = QLabel()
       predator_energy_label = QLabel()
       predator_max_label = QLabel()
       predator_mutation_label = QLabel()
       predator_attack_label = QLabel()
       
       infectivity_label = QLabel()
       lethality_label = QLabel()


       #telling the user what value they are changing
       prey_speed_title = QLabel("Average Speed")
       prey_population_title = QLabel("Starting population")
       prey_energy_title = QLabel("Average energy use")
       prey_max_energy_title = QLabel("Maximum energy")
       prey_mutation_title = QLabel("Chance of mutation")
       prey_attack_title = QLabel("Attack power")

       predator_speed_title = QLabel("Average Speed")
       predator_population_title = QLabel("Starting population")
       predator_energy_title = QLabel("Average energy use")
       predator_max_energy_title = QLabel("Maximum energy")
       predator_mutation_title = QLabel("Chance of mutation")
       predator_attack_title = QLabel("Attack power")

       infectivity_title = QLabel("Infectivity")
       lethality_title = QLabel("Lethality")
       species_affected = QLabel("Species affected")
       #getting the actual value of each slider
       self.prey_speed = self.prey_speed_slider.value()
       self.prey_avg_energy = self.prey_energy_slider.value()
       self.prey_mutation = self.prey_mutation_slider.value()
       self.prey_max_energy = self.prey_max_energy_slider.value()
       self.prey_population = self.prey_population_slider.value()
       self.prey_attack = self.prey_attack_slider.value()

       self.predator_speed = self.predator_speed_slider.value()
       self.predator_avg_energy = self.predator_energy_slider.value()
       self.predator_mutation = self.predator_mutation_slider.value()
       self.predator_max_energy = self.predator_max_energy_slider.value()
       self.predator_population = self.predator_population_slider.value()
       self.predator_attack = self.predator_attack_slider.value()

       #putting the values into a list to use for methods
       self.prey_values = [self.prey_speed,self.prey_population,self.prey_avg_energy,self.prey_max_energy,self.prey_mutation,self.prey_attack]
       self.predator_values = [self.predator_speed,self.predator_population,self.predator_avg_energy,self.predator_max_energy,self.predator_mutation,self.predator_attack]
       #grouping the slider, label and title together so they can be displayed properly within a method.
       self.prey_settings = [(self.prey_speed_slider,prey_speed_label,prey_speed_title), (self.prey_population_slider,prey_population_label,prey_population_title), (self.prey_energy_slider,prey_energy_label,prey_energy_title),(self.prey_max_energy_slider,prey_max_label,prey_max_energy_title), (self.prey_mutation_slider,prey_mutation_label,prey_mutation_title),(self.prey_attack_slider,prey_attack_label,prey_attack_title)]
       self.predator_settings = [(self.predator_speed_slider,predator_speed_label, predator_speed_title), (self.predator_population_slider,predator_population_label,predator_population_title), (self.predator_energy_slider,predator_energy_label,predator_energy_title),(self.predator_max_energy_slider,predator_max_label,predator_max_energy_title),(self.predator_mutation_slider,predator_mutation_label,predator_mutation_title),(self.predator_attack_slider,predator_attack_label,predator_attack_title)]
       disease_settings = [(self.infectivity_slider, infectivity_label, infectivity_title), (self.lethality_slider, lethality_label, lethality_title)]


       
       #this is the minimum and maximum values that the slider can be set to. It will go up in 5s.
       for setting in self.prey_settings + self.predator_settings + disease_settings :
          slider = setting[0]
          label = setting[1]
          stat = setting[2]
          if slider == self.prey_speed_slider or slider == self.prey_population_slider or slider == self.predator_speed_slider or slider == self.predator_population_slider:
             slider.setMaximum(25)
          else:
             
            slider.setMaximum(100)
          slider.setTickPosition(QSlider.TickPosition.TicksBelow)
          slider.setTickInterval(5)
          #here we make sure that the correct label is under the right slider
          slider.valueChanged.connect(lambda value, label = label, stat = stat.text(): self.display(label, value, stat))
        

       
       #here we add the slider and labels to the different layout
       
       

       
       
       for setting in self.prey_settings:
          prey_layout.addWidget(setting[2])
          prey_layout.addWidget(setting[0])
          prey_layout.addWidget(setting[1])
       
       for setting in self.predator_settings:
          predator_layout.addWidget(setting[2])
          predator_layout.addWidget(setting[0])
          predator_layout.addWidget(setting[1])


       for setting in disease_settings:
          disease_layout.addWidget(setting[2])
          disease_layout.addWidget(setting[0])
          disease_layout.addWidget(setting[1])
      
      #these have to be add separately as
       disease_layout.addWidget(species_affected)
       disease_layout.addWidget(self.prey_affected)
       disease_layout.addWidget(self.predator_affected)
       


       #setting the layout so each box will display things vertically
       prey_box.setLayout(prey_layout)
       predator_box.setLayout(predator_layout)
       disease_box.setLayout(disease_layout)
       
      #making sure that the boxes are horizontally in line
       box_layout.addWidget(prey_box)
       box_layout.addWidget(predator_box)
       box_layout.addWidget(disease_box)
      #adding everything to the main page so it is displayed in a stack
       main_layout.addWidget(title)
       main_layout.addLayout(box_layout)
       main_layout.addWidget(start_button)
       main_layout.setAlignment(start_button,Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
       window = QWidget()
       window.setLayout(main_layout)
       self.setCentralWidget(window)

       #def __init__(self,prey_speed,prey_avg_energy,prey_mutation,prey_max_energy,prey_population,prey_attack):


   #this will display the number that the slider is at    
    def display(self,label, value, stat):
        label.setText(f"{stat}: {value}")
        label.adjustSize()  # Expands label size as numbers get larger
  
    #when this function is called, it updates the value of the slider
    def update_value(self):
      #this will iterate through the sliders and the values at the same time
       for i in range(len(self.prey_settings)):
         prey_slider = self.prey_settings[i][0]
         predator_slider = self.predator_settings[i][0]
         self.prey_values[i] = prey_slider.value()
         self.predator_values[i] = predator_slider.value()
         #now we make sure that the variables themselves are updated, rather than just the list
         #self.prey_values = [self.prey_speed,self.prey_population,self.prey_avg_energy,self.prey_max_energy,self.prey_mutation,self.prey_attack]
         self.prey_speed,self.prey_population,self.prey_avg_energy,self.prey_max_energy,self.prey_mutation,self.prey_attack = self.prey_values
         self.predator_speed,self.predator_population,self.predator_avg_energy,self.predator_max_energy,self.predator_mutation,self.predator_attack = self.predator_values
         #this will also update the values for the simulation window
      

      
#this will start the simulation once the start button is pressed
    def start_sim(self):
        #similar to last time, this is instantiated to display the simulation once the button is pressed
       self.simulation_window = sim_window(self.prey_speed,self.prey_avg_energy,self.prey_mutation,self.prey_max_energy,self.prey_population,self.prey_attack,self.predator_speed,self.predator_avg_energy,self.predator_mutation,self.predator_max_energy,self.predator_population,self.predator_attack)
       if self.simulation_window.isVisible():
          self.simulation_window.hide()
       else:
          self.simulation_window.show()




#this is the main window where the simulation will be displayed
class sim_window(QWidget):
   def __init__(self,prey_speed,prey_avg_energy,prey_mutation,prey_max_energy,prey_population,prey_attack,predator_speed,predator_avg_energy,predator_mutation,predator_max_energy,predator_population,predator_attack):
      #def __init__(self,speed,max_energy,energy_use,attack,gen):
      super().__init__()
      self.setWindowTitle("Simulation")
      #this is the window where the simulation will take place
      self.scene = QGraphicsScene(0,0,900,600)
      #we set the height and width of the screen
      self.WIDTH = 900
      self.HEIGHT = 600
      #this check whether the simulation is paused or not.
      self.paused = False 

      self.prey_speed = prey_speed
      self.prey_avg_energy = prey_avg_energy
      self.prey_mutation = prey_mutation
      self.prey_max_energy = prey_max_energy
      self.prey_attack = prey_attack
      self.prey_population = prey_population

      self.predator_speed = predator_speed
      self.predator_avg_energy = predator_avg_energy
      self.predator_mutation = predator_mutation
      self.predator_max_energy = predator_max_energy
      self.predator_attack = predator_attack
      self.predator_population = predator_population

      #now we can instantiate the amount of prey required
      self.prey_group = [Prey(self.prey_speed,self.prey_max_energy,self.prey_avg_energy,self.prey_attack,0,self.prey_mutation) for i in range(self.prey_population)]
      #def __init__(self,speed,max_energy,energy_use,attack,gen,mutation):
      for prey in self.prey_group:
         #spawn the prey at random positions
         x = randint(0,self.WIDTH-200)
         y = randint(0,self.HEIGHT-200)
         prey.setPos(100.0,300.0)
         self.scene.addItem(prey)
         prey.add_rays()
      
      #now adding the amount of predators needed
      self.predator_group = [Predator(self.predator_speed,self.predator_max_energy,self.predator_avg_energy,self.predator_attack,0,self.predator_mutation) for i in range(self.predator_population)]
      for predator in self.predator_group:
         #spawn the predators at random locations
         x = randint(200,self.WIDTH)
         y = randint(200,self.HEIGHT)
         predator.setPos(0,50)
         self.scene.addItem(predator)
         predator.add_rays()
      
      #adding some amount of food every so often
      self.food_list = []
      self.food_spawner = QTimer(self)
      self.food_spawner.timeout.connect(self.spawn_food)
      self.food_spawner.start(1000)

      self.food_despawner = QTimer(self)
      self.food_despawner.timeout.connect(self.spoil)
      self.food_despawner.start(1000)
      
        #I think it will be best to put the graph in this file, rather than have a separate file.
      self.graph = pg.PlotWidget()
      self.graph.setBackground('w') #setting a white background for the graph
      self.prey_pen = pg.mkPen(0,255,0) #this should make the pen colour green
      self.predator_pen = pg.mkPen(255,0,0)
      #adding the titles
      self.graph.setTitle("Population vs time")
      self.graph.setLabel("left","Population")
      self.graph.setLabel("bottom","time (seconds)")
      self.graph.addLegend()
      self.graph.setYRange(0,25) #as 25 should be the maximum population the prey and predators can have, that will be the limit
      self.seconds = [0,1]
      self.prey_num = [len(self.prey_group),len(self.prey_group)]
      self.predator_num = [len(self.predator_group),len(self.predator_group)]
      #this is what we will plot on the graph
      self.prey_line = self.graph.plot(
         self.seconds,
         self.prey_num,
         name = 'Prey population',
         pen = self.prey_pen

      )

      self.predator_line = self.graph.plot(
         self.seconds,
         self.predator_num,
         name = 'Predator population',
         pen = self.predator_pen
         
      )
      #adding a timer to periodically update the graph
      self.update_timer = QTimer()
      self.update_timer.timeout.connect(self.update_graph)
      self.update_timer.start(1000)

      self.graph.setFixedSize(400,300)
      

      view = QGraphicsView(self.scene)
      layout = QHBoxLayout()
      layout.addWidget(view)
      layout.addWidget(self.graph)
      self.graph.move(self.WIDTH-400,0)
      self.setLayout(layout)

      #now we add a timer so the simulation can update
      self.timer = QTimer(self)
      self.timer.timeout.connect(self.prey_loop)
      self.timer.timeout.connect(self.predator_loop)
      self.timer.start(80) #this will run at around 13 fps
    #here we make a loop of what the prey will do as defined by the flowchart that I made

     

   #method to update the graph every second
   def update_graph(self):
      #the maximum range of seconds that it will store is 20
      #first it checks if the seconds list only has 1 item
      if len(self.seconds) == 1:
         #if it does, then the list adds 1 on the end
         self.seconds = [0,1]
         #if it's between 1 and 20, then it will append the last number in the list plus one
      elif len(self.seconds) > 1 and len(self.seconds) <= 20:
         self.seconds.append(self.seconds[-1]+ 1)
      else:
         #the same thing happens here, but remove the first item in the list.
         self.seconds = self.seconds[1:]
         self.seconds.append(self.seconds[-1]+ 1)
      
      #a similar update occurs to prey and predator population every second
      if len(self.prey_num) <= 20:
         self.prey_num.append(len(self.prey_group))
      else:
         self.prey_num = self.prey_num[1:]
         self.prey_num.append(len(self.prey_group))

      if len(self.predator_num) <= 20:
         self.predator_num.append(len(self.predator_group))
      else:
         self.predator_num = self.predator_num[1:]
         self.predator_num.append(len(self.predator_group))
      
      self.prey_line.setData(self.seconds,self.prey_num)
      self.predator_line.setData(self.seconds,self.predator_num)
   
   def prey_loop(self):
      prey_list = self.prey_group.copy()
      for prey in prey_list:
         prey.random_move(0,self.HEIGHT,0,self.WIDTH)
         prey.eat(self.scene,self.food_list)
         #prey.die(self.scene,self.prey_group)
         #prey.reproduce(self.scene,self.prey_group)
         #prey.defend(self.scene)


   def predator_loop(self):
      predator_list = self.predator_group.copy()
      for predator in predator_list:
         predator.detect()
         predator.move(0,self.HEIGHT,0,self.WIDTH)
         #predator.fight(self.scene)
         #predator.eat(self.scene)

         
   def spawn_food(self):
      food = Food(self,self.scene,self.food_list)
      dead_prey = Dead_prey(30,QPointF(1,2))
      #x = randint(0,self.WIDTH)
      #y = randint(0,self.HEIGHT)
      x = randint(0,50)
      y = randint(0,50)
      food.setPos(x,y)
      dead_prey.setPos(x,y)
      self.scene.addItem(food)
      self.scene.addItem(dead_prey)
      self.food_list.append(food)

   def spoil(self):
      food_group = self.food_list.copy()
      for food in food_group:
         food.lose_energy()
         food.spoil(self.scene,self.food_list)

      
   #this should check if the spacebar is pressed during the simulation. If it is, then it should pause the simulation.
   def keyPressEvent(self, event: QKeyEvent):
      if event.key() == Qt.Key.Key_Space:
         print('spacebar pressed')
         if self.paused is False:
            self.paused = True
            self.timer.stop()
            self.update_timer.stop()
            self.food_spawner.stop()
         elif self.paused is True:
            self.paused = False
            self.timer.start()
            self.update_timer.start()
            self.food_spawner.stop()
