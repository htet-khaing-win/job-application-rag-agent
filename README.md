# Job Application RAG agent

> Multi-agent RAG system that generates personalized, ATS-optimized cover letters by intelligently matching candidate resumes to job descriptions using vector search, LLM orchestration, and privacy-first design.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://langchain.com/langgraph)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-orange.svg)](https://www.pinecone.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Safety & Privacy Mechanisms](#safety--privacy-mechanisms)
- [Technical Highlights](#technical-highlights)
- [Installation Guide](#installation-guide)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

I hate having to do repetitive work. So I created this workflow.

### **The Problem**
- Generic cover letters get rejected by ATS systems
- Manual customization is time-consuming and inconsistent
- AI-generated letters often contain hallucinated qualifications
- Privacy concerns with uploading sensitive resume data

### **The Solution**
A stateful, multi-agent workflow that:
1. **Validates** job descriptions for authenticity
2. **Researches** company information using real-time web search
3. **Retrieves** relevant candidate qualifications via semantic vector search
4. **Verifies** every claim against source documents (anti-hallucination layer)
5. **Generates** personalized cover letters with ATS keyword optimization
6. **Critiques & Refines** output through iterative quality gates

---

## ✨ Key Features

### 🔒 **Privacy-First Architecture**
- **Presidio PII Guard Integration**: Automatically redacts personally identifiable information (names, emails, phone numbers, LinkedIn/GitHub URLs, locations, certificates) before vectorization
- **Custom Burmese Name Recognition**: Extended NER model to support multi-cultural name patterns
- **Local Processing**: All sensitive data processing happens on-device using Ollama LLMs

### 🧠 **Intelligent Document Processing**
- **Multi-Format Support**: PDF, DOCX, TXT resume parsing with `pdfplumber` and `python-docx`
- **Semantic Chunking**: Uses `SemanticChunker` from LangChain to split documents at natural conceptual boundaries (vs. arbitrary character limits)
- **Namespace Isolation**: Pinecone namespaces ensure complete separation between candidate resumes (prevents cross-contamination)

### 🎭 **Multi-Agent Orchestration**
- **Dual-LLM Strategy**: 
  - **Generator** (Mistral 7B): Creative, fluent text generation
  - **Critic** (Qwen2.5 7B): Analytical grading and verification
- **Stateful Graph Workflow**: LangGraph manages complex conditional routing and retry logic
- **Iterative Refinement**: Up to 3 critique-refinement loops to polish output quality

### 🌐 **Real-Time Company Research**
- **Tavily AI Integration**: Fetches current company mission, values, and recent news
- **Contextual Personalization**: Opening paragraphs reference actual company initiatives (not generic "I'm excited about [Company]" statements)

### 🛡️ **Anti-Hallucination Safeguards**
- **Claim Verification Node**: Cross-references every statement in the candidate summary against retrieved resume chunks
- **Evidence Tracing**: Requires LLM to cite specific source quotes for each qualification
- **Minimum Confidence Threshold**: Rejects summaries where <70% of claims are verifiable

---

## 🏗️ System Architecture

### **High-Level Flow Diagram**

```
<img width="510" height="943" alt="graph_flow" src="https://github.com/user-attachments/assets/54ad50c0-0f18-4fa4-933d-1618d50ddbdb" />

```

### **Component Breakdown**

#### **1. Ingestion Pipeline** (`database.py`)
```
User Upload → Format Detection → Text Extraction → PII Redaction → 
Semantic Chunking → Nomic Embeddings → Pinecone Upsert
```

#### **2. LangGraph Workflow** (`graph.py`)

**Conditional Routing:**
- **Query Rewriting**: If retrieval score < 60% (max 2 attempts)
- **Fallback Handling**: Invalid JD, empty database, or severe mismatch
- **Refinement Loop**: Up to 3 critique-driven rewrites

#### **3. Privacy Layer** (`privacy.py`)

**Presidio Configuration:**
- **NLP Engine**: spaCy's `en_core_web_trf` (transformer-based NER)
- **Custom Recognizers**:
  - LinkedIn/GitHub URL patterns
  - Phone number formats (international support)
  - Certificate URLs (Credly, Coursera, badges.alignment.org)
  - Burmese name dictionary (50+ common components)


---

## 🛡️ Safety & Privacy Mechanisms

### **1. PII Protection**
### **2. Hallucination Prevention**
### **3. Error Handling**

---

## 🔧 Technical Highlights

- **Cross-Namespace Search**: Queries all uploaded resumes simultaneously
- **Score-Based Filtering**: Only high-confidence matches reach the LLM
- **Top-K Aggregation**: Combines best chunks across resumes (diversity)

### **Dual-LLM Orchestration**
```python
generator_llm = ChatOllama(model="mistral:7b", temperature=0.7)  # Creative
critic_llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)     # Analytical
```

**Role Specialization:**
| Agent | Model | Temperature | Purpose |
|-------|-------|-------------|---------|
| **Generator** | Mistral 7B | 0.7 | Fluent, natural cover letter prose |
| **Critic** | Qwen2.5 7B | 0.0 | Deterministic scoring and verification |

---

## 📦 Installation Guide

### **Prerequisites**
- Python
- [Ollama](https://ollama.ai/) installed and running
- Pinecone account (free tier sufficient)
- Tavily API key (free tier: 1000 requests/month)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/htet-khaing-win/job-application-rag-agent.git
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf  # Presidio NLP model
```

### **Step 3: Download Ollama Models**
```bash
ollama pull mistral:7b-instruct
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### **Step 4: Configure Environment Variables**
Create a `.env` file in the project root:
```env
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
```

**Get Your API Keys:**
- **Pinecone**: https://app.pinecone.io/ → "API Keys" → Create Key
- **Tavily**: https://app.tavily.com/ → Sign up → Copy API Key

### **Ingest Resume(s)**
```bash
python database.py /path/to/your/resume.pdf
```

**Supported Formats:**
- `.pdf` (multi-page support)
- `.docx` (Microsoft Word)
- `.txt` (plain text)

**Validation:**
- Max file size: 5MB
- Automatic PII redaction applied
- Duplicate filenames trigger cascading delete

---

## 💡 Usage Examples

### **Example 1: Personalized Cover Letter Generation**

```bash
python main.py
```

**Input (Paste when prompted):**
```
Senior Machine Learning Engineer
PooPooPeePee AI is seeking an ML Engineer with 5+ years of experience in production ML systems, 
proficiency in PyTorch/TensorFlow, and strong skills in MLOps. The ideal candidate 
has deployed models at scale and contributed to open-source ML projects.
DONE
```

**Output:**
```
Company name detected: PooPooPeePee AI
Researching company information...
Retrieved 5 relevant resume chunks (Match Score: 87.3%)

Your Cover Letter is Ready:
---------------------------------
Dear Hiring Manager,

I am writing to express my interest in the Senior Machine Learning Engineer 
position at Acme AI. Having followed your recent launch of the AutoML platform 
that democratizes model deployment for non-technical users, I am excited about 
the opportunity to contribute to your mission of making AI accessible.

In my previous role, I architected a real-time recommendation engine processing 
2M+ daily requests with 40ms p99 latency, leveraging PyTorch and Kubernetes-based 
model serving. This system increased user engagement by 23% while reducing 
infrastructure costs by 35% through model quantization and efficient batching...

[Rest of letter]
---------------------------------
 Retrieval Score: 87/100
 Refinement Iterations: 2
```

### **Example 2: Researching Company Background**

```bash
python main.py
```

**Input:**
```
We're looking for a DevOps Engineer with Kubernetes and Terraform experience...
DONE
```

**System Prompt:**
```
Please enter the company name: TechCorp
 Company set to: TechCorp
 Researching company information...
```

### **Example 3: Low Match Scenario (Fallback)**

**Job Description:** "Seeking a Registered Nurse with ICU experience..."

**Output:**
```
I couldn't find a strong match between your resumes and this job description.

Suggestions:
- Upload a more relevant resume for this role
- Ensure the job description is complete and specific
- Try a different position that better matches your background

Would you like to try a different job description?
```

---

## 📁 Project Structure

```
job-application-rag-agent/
│
├── main.py                 # Entry point and orchestration
├── graph.py                # LangGraph workflow definition
├── node.py                 # Individual agent nodes (ingest, generate, critique, etc.)
├── state.py                # Pydantic state model
├── database.py             # Pinecone operations and resume ingestion
├── privacy.py              # Presidio PII Guard configuration
│
├── requirements.txt        # Python dependencies
├── .env                    # Template for environment variables
├── README.md               # This file
│
├── graph.png               # LangGraph visualization (generated)
└── resumes/                # Sample test resumes (add to gitignore)
```

### **Scalability**
- **Concurrent Users**: Single-user CLI 
- **Resume Limit**: No hard cap 

---

## Future Enhancements

### **Planned Features**
- [ ] **Web Interface**: Frontend for non-technical users
- [ ] **Hybrid Search**: Combine vector search with keyword filters
- [ ] **Caching Layer**: Redis for frequently requested company research

---

### **Development Setup**
```bash
pip install -r requirements.txt        # Install Dependencies
python database.py <Your Resume File>  # Add Resume to Pinecone Vector DB
pytest tests/                          # Run test suite
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

