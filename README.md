# **Website Chatbot using Hugging Face**

## 📌 Overview
This project is a console-based chatbot that interacts with a website.  
It scrapes website content, processes it, and answers user questions using Hugging Face models.

---

## 🚀 Features
- Web scraping using BeautifulSoup  
- Cleaned and structured website data  
- Context-based chatbot (answers only from website content)  
- Uses Hugging Face Inference API  
- Console-based interaction  
- Supports both API and local model  

---

## 🛠️ Tech Stack
- Python  
- BeautifulSoup  
- Hugging Face (Inference API)  
- Transformers  
- Pickle  

---

## 📂 Project Structure
- `web_scrape.py` → Scrapes website data  
- `chatbot.py` → Main chatbot  
- `data.pkl` → Stored website content  
- `.env` → API key (not uploaded)  

---

## ⚙️ Setup Instructions
1. Clone the repository  
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_api_key
   ```

---

## ▶️ How to Run
1. Scrape website:
   ```
   python web_scrape.py https://botpenguin.com/
   ```
2. Run chatbot:
   ```
   python chatbot.py
   ```

---

## 💬 Example
```
You: What is this website?
Bot: This website provides AI chatbot solutions...
```

---

## 🎯 Objective
To build a chatbot that answers questions based only on website content using NLP models.

---

## ⚠️ Note
- `.env` file is not included for security reasons  
- API key must be added manually  
