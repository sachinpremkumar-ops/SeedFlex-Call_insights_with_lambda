# 🎯 Call Insights - AI-Powered Call Center Analytics

A **multi-agent system** built with **LangGraph** that processes call center audio files and extracts actionable insights. Transforms raw audio conversations into structured data including summaries, sentiment analysis, action items, and topic classification.

---

## 🏗️ System Architecture

This system implements a **sequential multi-agent workflow** that processes audio files through specialized AI agents:

```
Audio File → Ingestion → Speech   →    Analysis Agents → Storage & Insights
                ↓           ↓                 ↓
            File Mgmt   Transcription  Parallel Analysis
                            ↓                 ↓
                        Translation    Summarization
                                       Topic Classification
                                       Key Points Extraction
                                       Action Items
                                       Sentiment Analysis
```

### 🤖 Agent Pipeline

The system consists of **8 specialized agents** working in sequence:

| Agent | Description |
|-------|-------------|
| 🎯 **Ingestion Agent** | Manages file discovery and state transitions |
| 🎤 **Speech Agent** | Handles transcription and translation |
| 📝 **Summarization Agent** | Creates concise conversation summaries |
| 🏷️ **Topic Classification Agent** | Categorizes conversation topics |
| 🔑 **Key Points Agent** | Extracts main discussion points |
| ✅ **Action Items Agent** | Identifies actionable tasks |
| 😊 **Sentiment Analysis Agent** | Analyzes emotional tone |
| 💾 **Storage Agent** | Stores results and manages embeddings |

---

## 🚀 Key Features

- **🤖 Multi-Agent Architecture**: LangGraph-based workflow orchestration
- **🛠️ Tool-Based Design**: Well-structured tools with comprehensive error handling
- **📊 State Management**: Type-safe state management with TypedDict
- **☁️ AWS Integration**: S3, Secrets Manager, and RDS PostgreSQL
- **📈 LangSmith Integration**: Complete observability and tracing
- **🧩 Modular Design**: Clean separation of concerns
- **🔄 Error Recovery**: Automatic rollback and retry mechanisms
- **📊 Performance Monitoring**: Built-in tracking and metrics

---

## 📁 Project Structure

```
call-insights/                          
├── 📄 langgraph.json                   # LangGraph configuration
├── 📄 pyproject.toml                   # Project dependencies and metadata
├── 📄 requirements.txt                 # Python dependencies
├── 📄 uv.lock                          # UV lock file for reproducible builds
├── 📁 src/                             # Source code directory
│   ├── 📄 graph.py                     # Main workflow orchestrator (ACTIVE)
│   ├── 📄 graph2.py                    # Experimental workflow (COMMENTED)
│   ├── 📄 hubspot_test.py              # HubSpot API integration test
│   ├── 📁 Tools/                       # Agent-specific tools
│   │   ├── 📄 Ingestion_Agent_Tools.py      # File management tools
│   │   ├── 📄 Speech_Agent_Tools.py         # Transcription & translation
│   │   ├── 📄 Summarization_Agent_Tools.py  # Text summarization
│   │   ├── 📄 Topic_Classification_Agent_Tools.py # Topic categorization
│   │   ├── 📄 Key_Points_Agent_Tools.py     # Key points extraction
│   │   ├── 📄 Action_Items_Agent_Tools.py   # Action items identification
│   │   ├── 📄 Sentiment_Analysis_Agent.py   # Sentiment analysis
│   │   └── 📄 Storage_Agent_Tools.py        # Data storage & embeddings
│   ├── 📁 utils/                       # Utility modules
│   │   ├── 📄 prompt_templates.py      # AI agent prompts
│   │   ├── 📄 openai_utils.py          # OpenAI API utilities
│   │   ├── 📄 rds_utils.py             # Database connection utilities
│   │   └── 📄 s3_utils.py              # AWS S3 utilities
│   ├── 📁 sql/                         # Database schema and queries
│   │   └── 📄 tables_sql.py            # Database table definitions
│   ├── 📁 agents/                      # Additional agent modules (empty)
│   └── 📁 mcp/                         # Model Context Protocol (empty)
├── 📁 static/                          # Static web assets
├── 📁 tests/                           # Test files
└── 📁 logs/                            # Application logs
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **🐍 Python 3.12+**
- **☁️ AWS Account** with S3, Secrets Manager, and RDS access
- **🔑 OpenAI API Key**
- **📊 LangSmith Account**

### 🚀 Quick Start

```bash
# 1. Install UV (if not already installed)
pip install uv

# 2. Clone the repository
git clone https://0fmfrelo8dpli812oabsduzy-admin@bitbucket.org/seedflex/call-transcription-insights.git
cd call-transcription-insights
cd call-insights

# 3. Install dependencies
uv sync

# 4. Install project in editable mode (REQUIRED)
uv pip install -e .
```

---

## 🔍 LangSmith Setup & Configuration

LangSmith provides comprehensive observability for your LangGraph workflows:

### 1. Create LangSmith Account
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up for a free account
3. Create a new project 

### 2. Get API Key
1. Navigate to **Settings → API Keys**
2. Create a new API key
3. Copy the key to your `.env` file

### 3. Environment Variables
```bash
# Add these to your .env file
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=ls__your_api_key_here
LANGSMITH_PROJECT=<your-project-name>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=<your-project-name>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

```
---

## 🎯 Usage

### 🚀 Running with LangGraph CLI

```bash
# Start the LangGraph server
uv run langgraph dev

# Or use a custom port if default is occupied
uv run langgraph dev --port 3000
```

*Built using LangGraph, OpenAI, and AWS. Designed for production-scale call center analytics.*# SeedFlex-Call_insights_with_lambda
# SeedFlex-Call_insights_with_lambda
