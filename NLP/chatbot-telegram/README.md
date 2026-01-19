# 🤖 Telegram RAG Bot with Groq

A Retrieval-Augmented Generation (RAG) chatbot that runs on Telegram, powered by Groq's fast LLM API. The bot can answer questions based on your custom document knowledge base using semantic search and AI-generated responses.

## ✨ Features

- 🔍 **Semantic Search**: Uses sentence transformers to find relevant information
- 🚀 **Fast Responses**: Powered by Groq's lightning-fast LLM API
- 💬 **Telegram Integration**: Easy to use through Telegram messenger
- 📚 **Custom Knowledge Base**: Add your own documents and data
- 🎯 **RAG Pipeline**: Combines retrieval and generation for accurate answers
- 🆓 **Free to Run**: Works on Google Colab's free tier

## 🛠️ Tech Stack

- **Python 3.x**
- **sentence-transformers**: For document embeddings
- **Groq API**: For fast LLM inference
- **python-telegram-bot**: For Telegram integration
- **NumPy**: For vector similarity calculations

## 📋 Prerequisites

Before running the bot, you need:

1. **Telegram Bot Token**
   - Open Telegram and search for [@BotFather](https://t.me/botfather)
   - Send `/newbot` and follow the instructions
   - Copy the token (format: `TELEGRAM API TOKEN`)

2. **Groq API Key** (Free)
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up for a free account
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key (format: `gsk_...`)

## 🚀 Quick Start

### Option 1: Run on Google Colab (Recommended)

1. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Create a new notebook

2. **Paste the Code**
   - Copy the entire bot code into a single cell

3. **Add Your API Keys**
   ```python
   TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
   GROQ_API_KEY = "YOUR_GROQ_API_KEY"
   ```

4. **Run the Cell**
   - Press `Shift + Enter` or click the Play button
   - Wait for "Bot is now running!" message

5. **Test Your Bot**
   - Open Telegram
   - Search for your bot by username
   - Send `/start`
   - Try: `/ask what is the refund policy?`

### Option 2: Run Locally

1. **Install Dependencies**
   ```bash
   pip install python-telegram-bot sentence-transformers groq nest-asyncio numpy
   ```

2. **Save the Code**
   - Save the bot code as `telegram_bot.py`

3. **Add Your API Keys**
   - Edit the configuration section with your tokens

4. **Run the Bot**
   ```bash
   python telegram_bot.py
   ```

## 💡 Usage

### Available Commands

- `/start` - Welcome message and bot introduction
- `/help` - Display help and usage instructions
- `/ask <question>` - Ask the bot a question

### Example Questions

```
/ask what is the refund policy?
/ask how many leaves do employees get?
/ask how to make pancakes?
/ask is remote work allowed?
/ask do you provide tech support?
```

## 📚 Customizing the Knowledge Base

Edit the `docs` dictionary to add your own content:

```python
docs = {
    "policy": """Your company policy here...""",
    
    "faq": """Your FAQ content here...""",
    
    "recipes": """Your recipes or other content...""",
    
    "new_section": """Add more sections as needed..."""
}
```

### Tips for Better Results

- **Keep chunks manageable**: The code automatically splits text into 300-character chunks
- **Use clear formatting**: Well-structured content helps the bot find relevant information
- **Add context**: Include headers and clear labels in your documents
- **Test queries**: Try different phrasings to see what works best

## ⚙️ Configuration Options

### Adjust Retrieval Settings

```python
# In the retrieve() function
def retrieve(query, k=3):  # Change k to retrieve more/fewer chunks
```

### Modify LLM Settings

```python
# In the generate_answer() function
groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Try different models
    temperature=0.3,  # Lower = more focused, Higher = more creative
    max_tokens=500    # Maximum response length
)
```

### Available Groq Models

- `llama-3.3-70b-versatile` - Best balance (recommended)
- `llama-3.1-8b-instant` - Fastest responses
- `mixtral-8x7b-32768` - Large context window

## 🔧 Troubleshooting

### Bot Not Responding

1. **Check API Keys**: Ensure both tokens are correct
2. **Verify Bot Token**: Test with BotFather's `/token` command
3. **Check Groq Quota**: Free tier has rate limits (30 req/min)
4. **Review Logs**: Check Colab output for error messages

### Common Errors

**"Invalid token"**
- Double-check your Telegram bot token
- Make sure there are no extra spaces

**"API key not valid"**
- Verify your Groq API key is correct
- Check if your account is active

**"Bot timeout"**
- Restart the Colab cell
- Check your internet connection

## 📊 How It Works

1. **Embedding Creation**: Documents are converted to vector embeddings
2. **Query Processing**: User question is also embedded
3. **Similarity Search**: Top-3 most relevant chunks are retrieved
4. **Answer Generation**: Groq LLM generates answer based on context
5. **Response**: Answer is sent back via Telegram

## 🌟 Advanced Features

### Add Conversation History

The bot already tracks the last 5 interactions per user:

```python
history.setdefault(user, []).append((query, answer))
history[user] = history[user][-5:]
```

### Extend to Multiple File Types

Add support for PDFs, Word docs, etc.:

```python
# Install additional libraries
!pip install pypdf2 python-docx

# Add file processing functions
from PyPDF2 import PdfReader

def extract_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share improvements

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review Groq API documentation: [console.groq.com/docs](https://console.groq.com/docs)
3. Check python-telegram-bot docs: [docs.python-telegram-bot.org](https://docs.python-telegram-bot.org/)

## 🎯 Future Enhancements

- [ ] Add support for file uploads
- [ ] Implement multi-language support
- [ ] Add admin commands for managing documents
- [ ] Create web interface for document management
- [ ] Add analytics and usage tracking
- [ ] Support for images and multimedia

## 🙏 Acknowledgments

- [Groq](https://groq.com) for fast LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [python-telegram-bot](https://python-telegram-bot.org/) for Telegram integration

---

**Happy Chatting! 🚀**
