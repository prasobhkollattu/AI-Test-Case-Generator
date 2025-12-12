# 🤖 AI Test Case Generator

**Automatically generate comprehensive test cases using AI**

---

## 🎯 Features

- ✅ **Add Training Examples** via simple web interface
- ✅ **Auto-Training** in background (no waiting!)
- ✅ **Generate Test Cases** instantly for any feature
- ✅ **No Database Required** - everything in memory
- ✅ **Simple Setup** - just 2 commands

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Backend
```bash
cd backend
python app.py
```

Server runs on: http://localhost:8000

### 3. Open Frontend

Open `frontend/index.html` in your web browser

---

## 📖 How to Use

### Step 1: Add Training Examples

1. Go to **"Training Data"** tab
2. Enter a feature description (e.g., "Login page")
3. Enter test cases for that feature
4. Click **"Add Training Example"**
5. Repeat 5-10 times with different features

### Step 2: Auto-Training

- After every 3 examples, model automatically trains in background
- You get instant response - no waiting!
- Training takes 20-40 minutes (happens in background)
- You can continue adding examples while training

### Step 3: Generate Test Cases

1. Go to **"Generate"** tab
2. Enter any feature description
3. Click **"Generate Test Cases"**
4. Get instant results!

---

## 💡 Example

**Input:**
```
Payment processing with credit card and PayPal
```

**Output:**
```
✅ POSITIVE TEST CASES
1. Verify payment with valid credit card
2. Verify PayPal payment succeeds
3. Verify payment confirmation email sent

❌ NEGATIVE TEST CASES
1. Verify error with expired card
2. Verify error with insufficient funds
3. Verify invalid CVV rejected

🔒 SECURITY TEST CASES
1. Verify card number encrypted
2. Verify CVV never stored
3. Verify PCI DSS compliance
```

---

## 🎓 Technical Details

- **Model**: GPT-2 fine-tuned on custom test cases
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Storage**: In-memory (no database)
- **Training**: Automatic background training

---

## 📊 System Architecture
```
User Browser (Frontend)
         ↓
    HTTP REST API
         ↓
FastAPI Backend (Python)
         ↓
GPT-2 Model (In-Memory)
```

---

## ⚙️ Configuration

In `backend/app.py`, you can adjust:
```python
AUTO_TRAIN_ENABLED = True           # Enable/disable auto-training
MIN_EXAMPLES_FOR_TRAINING = 5      # Minimum examples before training
RETRAIN_AFTER_N_EXAMPLES = 3       # Retrain after N new examples
```

---

## 📝 Notes

- **In-Memory Storage**: All data lost when server restarts (perfect for demo)
- **Auto-Training**: Happens automatically in background
- **GPU**: Will use GPU if available (much faster)
- **CPU**: Works on CPU (slower but functional)

---

## 🎯 Portfolio Highlights

This project demonstrates:
- ✅ Machine Learning / NLP
- ✅ API Development (FastAPI)
- ✅ Asynchronous Programming
- ✅ Full-Stack Development
- ✅ UI/UX Design
- ✅ Testing Domain Expertise

---

## 📧 Contact

[Your Wife's Name]  
Email: [email]  
LinkedIn: [linkedin]  
GitHub: [github]

---

**⭐ If you find this project interesting, please star it!**
