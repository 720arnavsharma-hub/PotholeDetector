\# 🚧 Pothole Detection System using YOLOv8



This project detects potholes in road images and videos using a trained YOLOv8 deep learning model.



Everything is included — just install dependencies and run.



---



\## 📦 Included



\- Trained model (best.pt)

\- Sample test files

\- Easy setup



---



\## ⚙ Requirements



\- Python 3.8+

\- pip



---



\## 🚀 Setup



```bash

python -m venv venv

venv\\Scripts\\activate

pip install ultralytics opencv-python

yolo detect predict model=best.pt source="D:\\aies project\\potholeimage1.png" 

