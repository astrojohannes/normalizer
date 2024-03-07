from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton, QCheckBox, QDialog, QComboBox
from PyQt5.QtCore import Qt


class AboutWin(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Store the main window reference
        self.initUI()

    def initUI(self):
        # Set window title and initial layout
        self.setWindowTitle('About')
        self.setGeometry(100, 100, 400, 260)  # x, y, width, height
        self.setFixedSize(400, 260)
        layout = QVBoxLayout()

        # Add text labels and make the website link clickable
        label_name = QLabel('Created by Johannes Puschnig, 2023-2024')
        label_name.setAlignment(Qt.AlignCenter)  # Center alignment
        label_web = QLabel('<a href="http://www.jpuschnig.com">Web: www.jpuschnig.com</a>')
        label_web.setOpenExternalLinks(True)  # Allow opening the link
        label_web.setAlignment(Qt.AlignCenter)
        label_contact = QLabel('Contact: johannes@jpuschnig.com')
        label_contact.setAlignment(Qt.AlignCenter)

        # Add close button
        button_close = QPushButton('Close')
        button_close.clicked.connect(self.close)  # Connect the button's clicked signal to close the window
        
        # Add widgets to layout
        layout.addWidget(label_name)
        layout.addWidget(label_web)
        layout.addWidget(label_contact)
        layout.addWidget(button_close)
        
        # Set the layout on the application's window
        self.setLayout(layout) 
