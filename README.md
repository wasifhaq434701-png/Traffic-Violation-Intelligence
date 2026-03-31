🚦 AI-Based Traffic Violation Detection System
![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/4.webp?raw=true)



![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/3.png?raw=true)

![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/5.jpg?raw=true)

📌 Overview
This project is an AI-powered traffic monitoring system that detects and analyzes multiple traffic violations in real-time using Computer Vision and Deep Learning.
The system processes images and videos to identify:
🚫 Helmet violations
👥 Triple riding on bikes
📱 Mobile phone usage while driving
🚗 Vehicle type classification
🔢 Number plate detection
It is designed to assist traffic authorities in automating enforcement and improving road safety.

🎯 Key Features
🔍 Real-time Object Detection
🪖 Helmet Detection
👨‍👨‍👦 Triple Riding Detection
📱 Mobile Phone Usage Detection
🚘 Vehicle Classification (Bike, Car, Truck, etc.)
🔢 Automatic Number Plate Recognition (ANPR)
🖼️ Violation Screenshot Storage
🗂️ Database Logging of Violations
🎥 Works on Both Images and Videos

🧠 System Architecture




Workflow:
Input Image/Video
Frame Extraction (for video)
Object Detection Model (YOLO / CNN-based)
Violation Detection Logic
Number Plate Recognition
Store Results (Images + Data)
Display Output

🛠️ Tech Stack
💻 Programming
Python 🐍
🧠 AI/ML Libraries
OpenCV
TensorFlow / PyTorch
YOLO (You Only Look Once)
📊 Data Handling
NumPy
Pandas
🗄️ Database
SQLite / Local Storage 

📸 Sample Outputs


![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/1.png?raw=true)

![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/6.jpg?raw=true)

![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/7.jpeg?raw=true)


Bounding boxes around detected vehicles
Labels like:
❌ No Helmet
⚠️ Triple Riding
📱 Mobile Usage
Vehicle type displayed
Number plate extracted

⚙️ Installation & Setup
# Clone the repository
git clone (https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence.git)

# Navigate to project folder
cd traffic-violation-detection

# Install dependencies
pip install -r requirements.txt

# Run the project
python RMain.py

📂 Project Structure
📁 traffic-violation-detection
│── 📁 models/                # Trained models
│── 📁 dataset/               # Dataset files
│── 📁 outputs/               # Detected violations images
│── 📁 database/              # Stored violation records
│── RMain.py                 # Main execution file
│── utils.py                 # Helper functions
│── requirements.txt
│── README.md

📊 Performance & Results


![image alt](https://github.com/wasifhaq434701-png/Traffic-Violation-Intelligence/blob/main/9.png?raw=true)


![image alt](https://raw.githubusercontent.com/wasifhaq434701-png/Traffic-Violation-Intelligence/refs/heads/main/2.avif)



High accuracy in detecting:
Helmet violations
Vehicle types
Real-time processing capability
Scalable for smart city applications

🚧 Challenges Faced
Detecting helmets under different lighting conditions
Differentiating between similar vehicle types (e.g., van vs truck)
Handling occlusions in crowded traffic
Improving mobile phone detection accuracy

🔮 Future Improvements
🌐 Web dashboard for monitoring violations
📲 Mobile app integration
☁️ Cloud deployment
🔔 Real-time alerts to authorities
🧠 Advanced AI models for better accuracy

🤝 Contribution
Contributions are welcome! Feel free to fork this repo and submit a pull request.

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
MOHAMMED WASIF UL HAQ
LinkedIn: www.linkedin.com/in/mohammedwasifulhaq
Github: https://github.com/wasifhaq434701-png?tab=overview&from=2026-02-01&to=2026-02-28

ABDUL SAMAD
LinkedIn: https://www.linkedin.com/in/abdul-samad-ab5a58316?utm_source=share_via&utm_content=profile&utm_medium=member_android
Github: https://github.com/AbdulSamad502


⭐ Support
If you like this project, don’t forget to star ⭐ the repository!

