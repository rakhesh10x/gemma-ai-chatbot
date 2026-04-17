# 🤖 Gemma 2B AI Chatbot

A lightweight AI chatbot built using Google's Gemma 2B models with an interactive Gradio interface. This project allows users to switch between base and instruction-tuned models and generate responses dynamically.

---

## 🚀 Features

* 🔄 Switch between **Base** and **Instruction-tuned** models
* 💬 Interactive chatbot UI using Gradio
* ⚡ Optimized for CPU and GPU environments
* 🔐 Secure Hugging Face authentication
* 🌐 Deployable on Hugging Face Spaces

---

## 🧠 Models Used

* google/gemma-2b (Base Model)
* google/gemma-2b-it (Instruction Model)

---

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Gradio
* Hugging Face Hub

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/Rakhesh143/gemma-2b-chatbot.git
cd gemma-2b-chatbot
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Login to Hugging Face

```
hf auth login
```

---

### 4️⃣ Run the application

```
python app.py
```

---

## 🌐 Deployment

Deployed using Hugging Face Spaces (Gradio SDK):

* Upload project files
* Add `HF_TOKEN` in Secrets
* Automatic build & deploy

---

## 📸 Screenshots

### 🖥️ Chatbot Interface (Instruction Model)

![UI 1](images/ui1.png)

---

### 🧠 Response Generation Example

![UI 2](images/ui2.png)

---

### 🔄 Model Switching (Base Model)

![UI 3](images/ui3.png)

---

## ⚠️ Challenges Faced

* Handling gated model access (403/401 errors)
* Authentication across local and Colab environments
* Slow inference on CPU
* Meta tensor / device issues

---

## 💡 Optimizations

* Reduced `max_new_tokens` for faster inference
* Adjusted `temperature` and `top_p`
* Used CPU-safe configurations
* Tested on Google Colab GPU for speed

---

## 🎯 Future Improvements

* Chat history support
* Streaming responses (typing effect)
* GPU-based deployment
* Faster lightweight models

---

## 🙌 Conclusion

This project demonstrates hands-on experience in building and deploying LLM-based applications using open-source models. It highlights understanding of model loading, inference, UI integration, and optimization.

---

## 📎 Author

**Rakesh Namineni**

* GitHub:  https://github.com/rakhesh10x
* LinkedIn: https://www.linkedin.com/in/rakhesh-namineni431

---

⭐ If you like this project, give it a star!
