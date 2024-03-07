from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont

class AboutWin(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Store the main window reference
        self.initUI()

    def initUI(self):
        # Set window title and initial layout
        self.setWindowTitle('About')
        self.setGeometry(100, 100, 560, 420)  # x, y, width, height
        self.setFixedSize(560, 420)
        main_layout = QVBoxLayout()  # The overall dialog layout

        # Define font
        font = QFont()
        font.setPointSize(18)

        # Create a horizontal layout for the logo
        logo_layout = QHBoxLayout()
        
        # Load and add the logo
        logo_label = QLabel(self)
        pixmap = QPixmap("Logo_spectrum_normalizer.png")
        logo_label.setPixmap(pixmap)
        logo_layout.addWidget(logo_label)

        # Add the logo layout to the main layout
        main_layout.addLayout(logo_layout)

        # Create a spacer that will push the rest to the bottom
        spacer_item = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addSpacerItem(spacer_item)

        # Create a vertical layout for the text and close button
        bottom_layout = QVBoxLayout()

        # Add text labels and make the website link clickable
        label_name = QLabel('Johannes Puschnig, 2023-2024')
        label_name.setAlignment(Qt.AlignLeft)  # Left alignment
        label_name.setFont(font)
        bottom_layout.addWidget(label_name)

        label_web = QLabel('<a href="http://www.jpuschnig.com">www.jpuschnig.com</a>')
        label_web.setFont(font)
        label_web.setOpenExternalLinks(True)  # Allow opening the link
        label_web.setAlignment(Qt.AlignLeft)
        bottom_layout.addWidget(label_web)

        label_contact = QLabel('Contact: johannes@jpuschnig.com')
        label_contact.setAlignment(Qt.AlignLeft)
        label_contact.setFont(font)
        bottom_layout.addWidget(label_contact)

        # Add close button
        button_close = QPushButton('Close')
        button_close.clicked.connect(self.close)
        bottom_layout.addWidget(button_close)

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)
        
        # Set the main layout on the application's window
        self.setLayout(main_layout)

