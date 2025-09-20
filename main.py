import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2

#Model ve sınıflar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Batarya","Biyolojik","Kahverengi-Cam","Karton","Kıyafet",
           "Yesil-Cam","Metal","Kağıt","Plastik","Ayakkabı","Çöp","Beyaz-Cam"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("garbage_resnet18.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Ana pencere
class RecyclingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻ Geri Dönüşüm Sınıflandırıcı")
        self.setGeometry(200, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f3e5f5;
                font-family: 'Segoe UI';
            }
            QPushButton {
                background-color: #ce93d8;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #ba68c8;
            }
            QLabel#title {
                color: #6a1b9a;
                font-size: 28px;
                font-weight: bold;
            }
            QLabel#result {
                color: #4a148c;
                font-size: 20px;
                font-weight: bold;
            }
            QFrame#card {
                background: white;
                border-radius: 25px;
                padding: 15px;
            }
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.cap = None

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Başlık
        title = QLabel("♻ Geri Dönüşüm Sınıflandırıcı")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Görsel alanı
        self.image_card = QFrame()
        self.image_card.setObjectName("card")
        image_layout = QVBoxLayout()
        self.image_label = QLabel("Görsel burada görünecek")
        self.image_label.setFixedSize(500, 350)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ce93d8; color:#9c27b0;")
        image_layout.addWidget(self.image_label)
        self.image_card.setLayout(image_layout)
        main_layout.addWidget(self.image_card)

        # Sonuç etiketi
        self.result_label = QLabel("Tahmin sonucu burada")
        self.result_label.setObjectName("result")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        # Butonlar
        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Fotoğraf Yükle")
        self.upload_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.upload_btn)

        self.capture_btn = QPushButton("📷 Tek Fotoğraf")
        self.capture_btn.clicked.connect(self.capture_image)
        btn_layout.addWidget(self.capture_btn)

        self.live_btn = QPushButton("🎥 Canlı Tespit Başlat")
        self.live_btn.clicked.connect(self.toggle_live)
        btn_layout.addWidget(self.live_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    # Fotoğraf yükle
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if fname:
            pix = QPixmap(fname).scaled(self.image_label.width(),
                                        self.image_label.height(),
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.predict(Image.open(fname).convert("RGB"))

    #Tek kare çek
    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save("capture_temp.jpg")
            pix = QPixmap("capture_temp.jpg").scaled(self.image_label.width(),
                                                     self.image_label.height(),
                                                     Qt.KeepAspectRatio,
                                                     Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.predict(img)

    # Canlı akışı başlat / durdur
    def toggle_live(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
            self.live_btn.setText("🎥 Canlı Tespit Başlat")
        else:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.live_btn.setText("⏹ Canlı Tespiti Durdur")

    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Model tahmini
        pil_img = Image.fromarray(img)
        self.predict(pil_img)

        # Qt'ye çevir
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    # Tahmin fonksiyonu
    def predict(self, pil_img):
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            _, pred = torch.max(out, 1)
        self.result_label.setText(f"Tahmin: {classes[pred.item()]}")

# Uygulama Başlat
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecyclingApp()
    window.show()
    sys.exit(app.exec_())
