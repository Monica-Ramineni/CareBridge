# Personal Health Copilot - Backend

FastAPI backend for the Personal Health Copilot application with real-time medical query processing and agent activity tracking.

## Features

- ✅ **FastAPI server** with medical query endpoints
- ✅ **WebSocket support** for real-time communication
- ✅ **Agent activity tracking** for frontend display
- ✅ **LangSmith integration** for monitoring
- ✅ **Medical tools integration** (RAG, web search, research)
- ✅ **CORS enabled** for frontend communication

## Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the environment template
cp env_template.txt .env

# Edit .env with your API keys
nano .env
```

Required API Keys:
- `OPENAI_API_KEY` - OpenAI API key
- `LANGCHAIN_API_KEY` - LangSmith API key
- `TAVILY_API_KEY` - Tavily Search API key
- `COHERE_API_KEY` - Cohere API key

### 3. Run the Server
```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check

### Chat
- `POST /chat` - Send medical queries
- `WebSocket /ws` - Real-time communication

### Metrics
- `GET /metrics` - System metrics

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Architecture

- **FastAPI** - Web framework
- **WebSockets** - Real-time communication
- **LangChain** - LLM orchestration
- **LangSmith** - Monitoring and tracing
- **BM25 + Contextual Compression** - Advanced retrieval

## Next Steps

1. Integrate with frontend (Next.js)
2. Add persistent storage for medical documents
3. Implement full agent workflow
4. Add authentication and user management 