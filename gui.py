from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator

import cv2
import os

import object_detection

initial_image_path = 'data/Test189x110/7351859.jpg'
initial_min_confidence = 50.0
initial_overlap_threshold = 0.3


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        widget = QWidget()
        layout = QVBoxLayout()

        hlayout0 = QHBoxLayout()
        self.image_label = QLabel(self)
        hlayout0.addItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Expanding))
        hlayout0.addWidget(self.image_label)
        hlayout0.addItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout.addLayout(hlayout0)

        vlayout = QVBoxLayout()

        hlayout1 = QHBoxLayout()
        self.prev_image_button = QPushButton("Previous", self)
        self.prev_image_button.clicked.connect(self.on_prev_button)
        self.prev_image_button.setEnabled(False)
        self.next_image_button = QPushButton("Next", self)
        self.next_image_button.setEnabled(False)
        self.next_image_button.clicked.connect(self.on_next_button)
        hlayout1.addItem(QSpacerItem(20, 30, QSizePolicy.Expanding, QSizePolicy.Minimum))
        hlayout1.addWidget(self.prev_image_button)
        hlayout1.addWidget(self.next_image_button)
        hlayout1.addItem(QSpacerItem(20, 30, QSizePolicy.Expanding, QSizePolicy.Minimum))

        hlayout2 = QHBoxLayout()
        file_name_label = QLabel('File name', self)
        self.file_name = QLineEdit(self)
        self.file_name.setReadOnly(True)
        open_image_button = QPushButton("Open", self)
        open_image_button.clicked.connect(self.on_open_image)
        toggle_rec_label = QLabel('Hide boxes', self)
        self.toggle_rec_check = QCheckBox(self)
        self.toggle_rec_check.clicked.connect(self.update_widgets)
        hlayout2.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        hlayout2.addWidget(file_name_label)
        hlayout2.addWidget(self.file_name)
        hlayout2.addWidget(open_image_button)
        hlayout2.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        hlayout2.addWidget(toggle_rec_label)
        hlayout2.addWidget(self.toggle_rec_check)
        hlayout2.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        hlayout3 = QHBoxLayout()
        conf_label = QLabel('Min confidence', self)
        self.conf_line = QLineEdit(conf_label)
        self.conf_line.setText(str(initial_min_confidence))
        self.conf_line.setFixedWidth(40)
        self.conf_line.editingFinished.connect(self.update_widgets)
        self.conf_line.setValidator(QDoubleValidator(0, 100, 2))
        threshold_label = QLabel('Overlap threshold', self)
        self.threshold_line = QLineEdit(threshold_label)
        self.threshold_line.setText(str(initial_overlap_threshold))
        self.threshold_line.setFixedWidth(40)
        self.threshold_line.editingFinished.connect(self.update_widgets)
        self.threshold_line.setValidator(QDoubleValidator(0, 1, 2))
        hlayout3.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        hlayout3.addWidget(conf_label)
        hlayout3.addWidget(self.conf_line)
        hlayout3.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))
        hlayout3.addWidget(threshold_label)
        hlayout3.addWidget(self.threshold_line)
        hlayout3.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        vlayout.addLayout(hlayout1)
        vlayout.addLayout(hlayout2)
        vlayout.addLayout(hlayout3)
        layout.addLayout(vlayout)
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.setGeometry(600, 200, 400, 300)
        self.setWindowTitle('Hand recognition')
        self.show()

        self.process_path(initial_image_path)

    def on_open_image(self):
        path = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.png *.jpg)')
        if path[0]:
            self.process_path(path[0])

    def on_prev_button(self):
        self.current_index -= 1
        self.update_widgets()

    def on_next_button(self):
        self.current_index += 1
        self.update_widgets()

    def display_image(self, path):
        image = cv2.imread(path)

        scale_X = 2.0
        scale_Y = 2.0

        if not self.toggle_rec_check.isChecked():
            boxes = object_detection.detect_objects(image, float(self.conf_line.text()),
                                                    float(self.threshold_line.text()))
            image = cv2.resize(image, (0, 0), fx=scale_X, fy=scale_Y)
            for x0, y0, x1, y1, confidence, class_num in boxes:
                x0, y0, x1, y1 = int(scale_X * x0), int(scale_Y * y0), \
                                 int(scale_X * x1), int(scale_Y * y1)
                color = None
                text = ''
                if class_num == 0:
                    color = (0, 255, 0)
                    text = 'left'
                elif class_num == 1:
                    color = (255, 0, 0)
                    text = 'right'
                elif class_num == 2:
                    color = (0, 0, 255)
                    text = 'bad'

                if color:
                    cv2.rectangle(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
                    cv2.putText(image, text + ' (' + str(confidence) + ')', (x0, y0 - 2), 1, 1, color, 1, cv2.LINE_AA)
        else:
            image = cv2.resize(image, (0, 0), fx=scale_X, fy=scale_Y)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def process_path(self, image_path):
        dir_path, file_name = os.path.split(image_path)

        self.image_list = sorted([dir_path + '/' + path for path in os.listdir(dir_path) if '.jpg' in path])
        self.current_index = self.image_list.index(image_path)

        self.update_widgets()

    def update_widgets(self):
        if self.current_index < 0:
            self.prev_image_button.setEnabled(False)
        else:
            self.prev_image_button.setEnabled(True)

        if self.current_index >= len(self.image_list):
            self.next_image_button.setEnabled(False)
        else:
            self.next_image_button.setEnabled(True)

        file_path = self.image_list[self.current_index]
        dir_path, file_name = os.path.split(file_path)

        self.display_image(file_path)
        self.file_name.setText(file_name)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
