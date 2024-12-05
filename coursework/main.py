#this file is where we will run the code from. I originally ran it from the GUI file, but to make things easier, I'm now running it here
from GUI import Start_Window
import sys
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
window = Start_Window()

window.show()
app.exec()