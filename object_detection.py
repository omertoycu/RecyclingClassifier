import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2

# ----------- Cihaz ve sınıflar -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Batarya","Biyolojik","Kahverengi-Cam","Karton","Kıyafet",
           "Yesil-Cam","Metal","Kağıt","Plastik","Ayakkabı","Çöp","Beyaz-Cam"]

# ----------- Modeli yükle -----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("garbage_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# ----------- Transform -----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------- Arayüz -----------
class RecyclingClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻ Recycling Classifier")
        self.setGeometry(100,100,650,550)
        self.setStyleSheet("background-color: #f5f5f5;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Fotoğraf Yükle Butonu
        self.upload_button = QPushButton("📁 Fotoğraf Yükle")
        self.upload_button.setFixedSize(200,50)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.upload_button.clicked.connect(self.load_image)
        layout.addWidget(self.upload_button)

        # Fotoğraf Çek Butonu
        self.capture_button = QPushButton("📷 Fotoğraf Çek")
        self.capture_button.setFixedSize(200,50)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #0b7dda; }
        """)
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        # Canlı Nesne Tespiti Butonu
        self.object_detect_button = QPushButton("📹 Canlı Nesne Tespiti")
        self.object_detect_button.setFixedSize(250,50)
        self.object_detect_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 16px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #e68a00; }
        """)
        self.object_detect_button.clicked.connect(self.live_object_detection)
        layout.addWidget(self.object_detect_button)

        # Resim Label
        self.image_label = QLabel()
        self.image_label.setFixedSize(400,300)
        self.image_label.setStyleSheet("border: 2px solid #ddd; background-color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Resim buraya gelecek")
        layout.addWidget(self.image_label)

        # Sonuç Label
        self.result_label = QLabel("Tahmin sonucu burada")
        self.result_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #333; margin-top: 20px;")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    # Fotoğraf yükleme fonksiyonu
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

            img = Image.open(file_name).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)

            self.result_label.setText(f"Tahmin: {classes[pred.item()]}")

    # Fotoğraf çekme fonksiyonu
    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)

            self.result_label.setText(f"Tahmin: {classes[pred.item()]}")

            # Arayüzde göster
            img.save("temp_capture.jpg")
            pixmap = QPixmap("temp_capture.jpg")
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    # Canlı nesne tespiti fonksiyonu
    def live_object_detection(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
            predicted_class = classes[pred.item()]

            # Basit obje detection: kutu ve label
            h, w, _ = frame.shape
            top_left = (int(w*0.2), int(h*0.2))
            bottom_right = (int(w*0.8), int(h*0.8))
            cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)
            cv2.putText(frame, predicted_class, (top_left[0], top_left[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Frame'i göster
            cv2.imshow("Canlı Nesne Tespiti", frame)

            # 'q' ile çıkış
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ----------- Uygulamayı başlat -----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecyclingClassifier()
    window.show()
    sys.exit(app.exec_())
