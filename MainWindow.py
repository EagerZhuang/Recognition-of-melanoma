# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  : dejahu
# @Email   : 1148392984@qq.com
# @File    : window.py
# @Software: PyCharm
# @Brief   : 图形化界面

import torch
import torch.nn as nn
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil
from torchvision import transforms


class MainWindow(QTabWidget):

    def __init__(self):
        super().__init__()
        # self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('The interface for testing model')

        self.to_predict_name = "images/melanoma_9605.jpeg"  # temp image
        self.class_names = ['benign', 'malignant']  # Class name
        self.resize(900, 700)
        self.initUI()

    # Initialization interface
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('Arial', 15)

        # Left side layout
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("Welcome to Use the Melanoma Recognition Program")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        # Right side layout
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" Upload Image ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" Start recognition ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' The Result: ')
        self.result = QLabel("Waiting for recognition")
        label_result.setFont(QFont('Arial', 16))
        self.result.setFont(QFont('Arial', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        self.addTab(main_widget, 'Main')
        self.setTabIcon(0, QIcon('images/主页面.png'))

    # Event Upload
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # File type
        img_name = openfile_name[0]  # Get the name of image
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # Move to the current directory
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # Open image
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (300, 300))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("Waiting for recognition")

    # Event To predict
    def predict_img(self):
        class_names = self.class_names
        # load model
        net = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 256, (3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),

            nn.Conv2d(256, 512, (2, 2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),

            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2)
        )
        model = torch.load("models/CNNTrainingForSGD.pt") # Gets the trained data
        net.load_state_dict(model["state_dict"])

        transform_valid = transforms.Compose([
            transforms.Resize((300, 300), interpolation=3),
            transforms.ToTensor()
        ]
        )
        img = Image.open('images/target.png')  # read image
        img_ = transform_valid(img).unsqueeze(0)  # expand the dimension
        outputs = net(img_)  # get result

        _, indices = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        perc = percentage[int(indices)].item()
        result = class_names[indices]
        print('predicted:', result)

        self.result.setText(result)  # return to interface

    # Event exit
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'Exit',
                                     "Confirm to exit?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
