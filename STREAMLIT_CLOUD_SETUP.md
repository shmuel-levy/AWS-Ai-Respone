# 🚀 Streamlit Cloud Setup Guide

## 🔑 Setting Up Environment Variables in Streamlit Cloud

### **Problem**: Gemini API Key works locally but not in cloud deployment

### **Solution**: Configure environment variables in Streamlit Cloud

---

## 📋 Step-by-Step Instructions

### **1. Access Streamlit Cloud Dashboard**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Find your app: `aws-ai-respone`

### **2. Configure Environment Variables**

1. **Click on your app** in the dashboard
2. **Click "Settings"** (gear icon) in the top right
3. **Scroll down to "Secrets"** section
4. **Click "Add secrets"**

### **3. Add Your Gemini API Key**

Add this to the secrets section:

```toml
GEMINI_API_KEY = "AIzaSyAtbzwNPFSSVJib2gDF6EcUuVsXqD30eRo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS = 18
RETRIEVAL_TOP_K = 3
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
```

### **4. Alternative: Use .streamlit/secrets.toml**

If the above doesn't work, create a file `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIzaSyAtbzwNPFSSVJib2gDF6EcUuVsXqD30eRo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS = 18
RETRIEVAL_TOP_K = 3
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
```

### **5. Update Requirements File**

Make sure your app uses `requirements-minimal.txt` instead of `requirements.txt`:

1. In Streamlit Cloud settings
2. Set "Requirements file" to: `requirements-minimal.txt`

---

## 🔧 Troubleshooting

### **If API Key Still Doesn't Work:**

1. **Check the logs** in Streamlit Cloud dashboard
2. **Verify the key format** - should be 39 characters
3. **Test the key locally** first
4. **Redeploy the app** after adding secrets

### **Common Issues:**

| Issue | Solution |
|-------|----------|
| **"API Key not found"** | Add to Streamlit Cloud secrets |
| **"Invalid API Key"** | Check key format and validity |
| **"RuntimeError"** | Use `requirements-minimal.txt` |
| **"ChromaDB error"** | System will use fallback automatically |

---

## 🧪 Testing the Fix

### **1. After adding secrets:**

1. **Redeploy the app** in Streamlit Cloud
2. **Wait for deployment** to complete
3. **Test the live demo**: https://documents-reader-ai.streamlit.app/

### **2. Test functionality:**

1. **Upload a document** (PDF or TXT)
2. **Ask a question** about the document
3. **Verify the answer** is based on document content

### **3. Expected behavior:**

- ✅ Document uploads successfully
- ✅ Questions get answered based on document
- ✅ No "API Key" errors
- ✅ Fallback system works (no ChromaDB errors)

---

## 📞 Support

If you still have issues:

1. **Check Streamlit Cloud logs** for specific error messages
2. **Verify your Gemini API key** is valid and has quota
3. **Test locally first** to ensure the key works
4. **Contact Streamlit support** if needed

---

## 🎯 Quick Fix Summary

**The main issue**: Streamlit Cloud doesn't have access to your local `.env` file.

**The solution**: Add environment variables through Streamlit Cloud's secrets management.

**Expected result**: Your app will work in the cloud just like it works locally!
