# BittuGPT - Comprehensive Codebase Documentation

This document provides a line-by-line and module-by-module technical reference for every file and function within the **BittuGPT** application.

---

## Table of Contents
1. [Project Directory Layout](#1-project-directory-layout)
2. [Module Reference: app.py](#2-module-reference-apppy)
3. [Module Reference: agent.py](#3-module-reference-agentpy)
4. [Module Reference: database.py](#4-module-reference-databasepy)
5. [Module Reference: rag.py](#5-module-reference-ragpy)
6. [Module Reference: tools.py](#6-module-reference-toolspy)
7. [Frontend Reference: templates/index.html](#7-frontend-reference-templatesindexhtml)
8. [Configuration & Deployment Files](#8-configuration--deployment-files)

---

## 1. Project Directory Layout

```text
BittuGPT/
├── .env                        # Local secret keys & environment variables (git-ignored)
├── .env.example                # Sample environment template
├── .gitignore                  # Git ignore rules for virtual environments, secrets, databases
├── Dockerfile                  # Container image specification (Python 3.11-slim)
├── LICENSE                     # Apache 2.0 open-source license
├── README.md                   # General project introduction & quickstart guide
├── SYSTEM_DESIGN.md            # System architecture, sequence diagrams & ER diagrams
├── CODE_DOCUMENTATION.md       # Detailed technical reference for all code modules
├── requirements.txt            # Python dependencies
├── agent.py                    # LangGraph workflow, model caching, and ReAct agent
├── app.py                      # FastAPI server, endpoints, SSE streaming generator
├── database.py                 # SQLAlchemy ORM, SQLite DB operations, session history & memory
├── rag.py                      # Multi-format document parser, embeddings & ChromaDB indexing
├── tools.py                    # LangChain tools (Tavily search, RAG, calculator, memory)
├── test.py                     # Minimal CLI testing script for LangGraph stream
├── .github/
│   └── workflows/
│       └── cicd.yaml           # GitHub Actions automated CI/CD pipeline
├── templates/
│   └── index.html              # ChatGPT-style web UI with streaming, dictation, and file upload
├── uploads/                    # Directory for user-uploaded documents
└── data/                       # Directory for SQLite database files
```

---

## 2. Module Reference: `app.py`

`app.py` serves as the primary ASGI entry point and web server. Built on FastAPI, it exposes API endpoints for the web UI, file uploads, conversation retrieval, and real-time Server-Sent Events (SSE) streaming.

### 2.1 Initialization & Global Setup
- `app = FastAPI()`: Initializes the FastAPI server.
- `templates = Jinja2Templates(directory="templates")`: Configures Jinja2 for rendering HTML.
- `init_db()`: Initializes the SQLite tables upon application startup.
- `uploads` & `data` directories are created automatically via `Path(...).mkdir(exist_ok=True)`.

### 2.2 Endpoints

#### `GET /`
- **Function**: `home(request: Request)`
- **Returns**: Renders `templates/index.html`.

#### `GET /conversations`
- **Function**: `conversations()`
- **Purpose**: Fetches the list of active user chat conversations ordered by recency.
- **Returns**: JSON array containing:
  - `thread_id`: UUID of the conversation session.
  - `title`: Short title (derived from the first message).
  - `created_at`: ISO timestamp.
  - `updated_at`: ISO timestamp of the last message.

#### `GET /history/{thread_id}`
- **Function**: `history(thread_id: str)`
- **Parameters**: `thread_id` (string path parameter).
- **Purpose**: Loads all historical messages for a specific conversation session.
- **Returns**: JSON object `{ "messages": [ { "role": "user"|"assistant", "content": "..." } ] }`.

#### `POST /upload`
- **Function**: `upload_document(file: UploadFile, thread_id: str)`
- **Form Data**:
  - `file`: Uploaded file binary (`.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.csv`).
  - `thread_id`: Current conversation thread ID.
- **Logic**:
  1. Validates file extension against allowed formats.
  2. Generates a unique UUID and sanitizes filename.
  3. Writes file to `uploads/{uuid}_{safe_filename}`.
  4. Calls `add_document_to_rag(...)` to parse, chunk, embed, and index into ChromaDB.
  5. Updates conversation title to "Uploaded document" if it is a new chat.
- **Status Codes**: `200` (Success), `400` (Unsupported file type), `500` (Extraction error).

#### `POST /chat/stream`
- **Function**: `chat_stream(request: Request)`
- **Request Body (JSON)**:
  - `message`: User input string.
  - `thread_id`: Active session ID.
  - `model`: Selected Gemini model (e.g., `gemini-2.5-flash`).
- **Response**: `StreamingResponse` using `text/event-stream` media type.
- **Core Streaming Mechanics**:
  - `event_generator()`: An asynchronous generator that calls `agent.stream(...)` with `stream_mode="messages"`.
  - Filters out raw tool JSON outputs and internal tool execution using `should_stream_chunk(...)`.
  - Extracts visible text tokens using `extract_text_from_chunk(...)`.
  - Yields SSE formatted payloads: `data: {"token": "..."}\n\n`.
  - Saves the accumulated `final_answer` to the SQLite database via `save_chat_message(...)`.
  - Dispatches termination packet `data: {"done": true}\n\n`.

### 2.3 Helper Functions

#### `sse_data(payload: dict) -> str`
- Serializes a dictionary into standard SSE format: `data: <json_string>\n\n`.

#### `should_stream_chunk(chunk, metadata: dict) -> bool`
- Determines whether a chunk yielded by LangGraph should be forwarded to the browser.
- **Returns `False`** if:
  - Node is an execution tool node (`langgraph_node` contains `"tool"`).
  - Chunk is an instance of `ToolMessage`.
  - Chunk contains `tool_calls` or `invalid_tool_calls`.
- **Returns `True`** only for actual AI assistant narrative tokens.

#### `extract_text_from_chunk(chunk) -> str`
- Extracts clean textual characters from various `AIMessageChunk` data structures (strings, list of dict tokens, or block elements).

---

## 3. Module Reference: `agent.py`

`agent.py` constructs and orchestrates the autonomous **ReAct Agent** using LangGraph.

### 3.1 Model Configuration
- `DEFAULT_MODEL`: Reads from `GEMINI_MODEL` or `GOOGLE_MODEL`, defaulting to `"gemini-2.5-flash"`.
- `ALLOWED_MODELS`: Whitelist of supported Gemini models:
  - `gemini-2.5-flash`
  - `gemini-2.5-pro`
  - `gemini-2.5-flash-lite`
  - `gemini-1.5-flash`
  - `gemini-1.5-pro`

### 3.2 System Prompt (`SYSTEM_PROMPT`)
Defines BittuGPT's persona, capabilities, and decision policies:
- Directs the assistant to behave like ChatGPT.
- Explicitly mandates web search (Tavily) for time-sensitive questions (news, current events, prices, etc.).
- Directs document queries to `search_uploaded_documents`.
- Directs memory queries to `remember_this` and `recall_memory`.
- Directs mathematical queries to `calculator`.

### 3.3 Core Functions

#### `normalize_model_name(model_name: str | None) -> str`
- Validates model names sent from the UI. If not present in `ALLOWED_MODELS`, cleanly falls back to `DEFAULT_MODEL`.

#### `build_agent(model_name: str)`
- Compiles the LangGraph state machine:
  1. Instantiates `ChatGoogleGenerativeAI` with the specified model, `temperature=0.3`, and `streaming=True`.
  2. Binds tools to the model via `llm.bind_tools(tools)`.
  3. Defines `chatbot_node(state)`: Injects `SystemMessage(SYSTEM_PROMPT)` and invokes the bound LLM.
  4. Defines `ToolNode(tools)`: Executes requested tools.
  5. Assembles `StateGraph(MessagesState)`:
     - Edge: `START -> chatbot`
     - Conditional Edge: `chatbot -> tools_condition` (routes to `tools` if tools are invoked, else exits).
     - Edge: `tools -> chatbot` (returns tool feedback to the LLM).
  6. Configures `SqliteSaver` checkpointer using `data/langgraph_checkpoints.sqlite` to preserve multi-turn memory across turns.
  7. Returns the compiled graph workflow.

#### `get_agent(model_name: str | None = None)`
- Implements an **in-memory cache pattern** (`_AGENT_CACHE`). If an agent for the requested model has already been built, returns the cached instance; otherwise compiles, caches, and returns it.

---

## 4. Module Reference: `database.py`

`database.py` manages relational persistence using SQLAlchemy and SQLite (`data/chatbot_memory.db`).

### 4.1 SQLAlchemy ORM Models

#### `Conversation`
- `id`: Integer Primary Key.
- `thread_id`: Unique session string (UUID).
- `title`: Short descriptive title for sidebar listing.
- `created_at`: UTC timestamp of creation.
- `updated_at`: UTC timestamp of last activity.

#### `ChatMessage`
- `id`: Integer Primary Key.
- `thread_id`: Foreign key reference to session thread.
- `role`: Message author (`"user"` or `"assistant"`).
- `content`: Complete text content.
- `created_at`: UTC timestamp.

#### `LongTermMemory`
- `id`: Integer Primary Key.
- `thread_id`: Foreign key reference to session thread.
- `memory`: Saved user preference, fact, or instruction.
- `created_at`: UTC timestamp.

### 4.2 Database Operations

- `init_db()`: Creates database tables if they do not already exist.
- `create_or_update_conversation(thread_id: str, first_message: str | None = None)`:
  - If thread does not exist: creates a new `Conversation` with a title derived from the first 40 characters of `first_message`.
  - If thread exists: refreshes `updated_at` to the current timestamp.
- `list_conversations()`: Returns all conversations ordered by `updated_at DESC`.
- `save_chat_message(thread_id: str, role: str, content: str)`: Inserts a new chat record and updates conversation activity.
- `get_chat_history(thread_id: str)`: Fetches all messages for a thread ordered chronologically by `created_at ASC`.
- `save_memory(thread_id: str, memory: str) -> str`: Persists a string into `LongTermMemory` and returns a success confirmation.
- `search_memory(thread_id: str, query: str) -> str`: Returns the latest 20 stored memory items for the current session.

---

## 5. Module Reference: `rag.py`

`rag.py` implements the document ingestion and Retrieval-Augmented Generation subsystem using ChromaDB and Google Gemini Embeddings.

### 5.1 Initialization
- `embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`: Gemini dense vector generator.
- `vectorstore = Chroma(...)`: Connects to local persistent ChromaDB at directory `chroma_db` under collection `agentic_chatbot_docs`.

### 5.2 Document Ingestion Pipeline

#### `read_file_text(file_path: str) -> str`
- Extracts raw text depending on file extension:
  - `.pdf`: Uses `pypdf.PdfReader` to extract text from all pages.
  - `.docx`: Uses `docx2txt.process`.
  - `.txt`, `.md`, `.py`, `.csv`: Uses native UTF-8 file reading with fallback error ignoring.

#### `add_document_to_rag(file_path: str, thread_id: str) -> dict`
- **Steps**:
  1. Extracts text with `read_file_text`.
  2. Splits text using `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)`.
  3. Wraps chunks into `Document` objects with metadata: `{"thread_id": thread_id, "source": filename}`.
  4. Embeds and adds documents to ChromaDB via `vectorstore.add_documents(docs)`.
- **Returns**: `{"filename": filename, "chunks": number_of_chunks}`.

### 5.3 Retrieval Pipeline

#### `retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str`
- Executes similarity search filtered strictly by session: `filter={"thread_id": thread_id}`.
- If no documents match, returns `"No relevant uploaded document content found."`.
- Formats top-k matching documents into citation blocks:
  ```text
  [Source 1: filename.pdf]
  <chunk content>
  ```

---

## 6. Module Reference: `tools.py`

`tools.py` defines the tool definitions provided to the LangGraph ReAct agent.

### 6.1 Session Thread Tracking
- `CURRENT_THREAD_ID`: Global variable representing the active session thread.
- `set_current_thread_id(thread_id: str)`: Called before invoking the agent to bind tools to the active conversation.

### 6.2 Defined Tools

1. **`calculator(expression: str) -> str`**:
   - Safely parses and evaluates basic mathematical expressions.
   - Restricts `__builtins__` to empty dict to block arbitrary code execution.
   - Provides access only to `math` module functions and built-in numerical functions (`abs`, `round`, `min`, `max`, `sum`).

2. **`search_uploaded_documents(query: str) -> str`**:
   - Queries `retrieve_from_rag(query, CURRENT_THREAD_ID)`.
   - Used when users ask about their uploaded PDFs, docs, or code.

3. **`remember_this(memory: str) -> str`**:
   - Saves facts to long-term memory via `save_memory(CURRENT_THREAD_ID, memory)`.

4. **`recall_memory(query: str) -> str`**:
   - Recalls saved facts and preferences via `search_memory(CURRENT_THREAD_ID, query)`.

5. **`web_search (TavilySearch)`**:
   - Powered by Tavily API with `max_results=5`, `topic="general"`, `search_depth="advanced"`.
   - Fetches live information from the internet for current events and real-time data.

---

## 7. Frontend Reference: `templates/index.html`

The frontend is a self-contained, responsive Single Page Application (SPA).

### 7.1 Visual Architecture & Styles
- Dark mode theme inspired by modern AI chat applications:
  - Background: `#212121`
  - Sidebar: `#171717` with `#2f2f2f` borders.
  - User Bubbles: Modern dark gray accent `#2f2f2f`.
  - Assistant Bubbles: Transparent, clean typography with high readability.
- Flexible responsive layout with mobile drawer support.

### 7.2 Core JavaScript Functions

- `generateUUID()`: Generates UUIDv4 for session identifiers stored in `localStorage`.
- `loadConversations()`: Fetches `/conversations` and populates the sidebar chat history.
- `loadConversation(thread_id)`: Swaps active session, updates `threadId`, and fetches `/history/{thread_id}`.
- `newChat()`: Resets `threadId`, clears the chat message stream, and re-displays starter cards.
- `sendMessage()`:
  - Validates user input.
  - Appends user bubble to UI.
  - Initiates `fetch("/chat/stream", { ... })`.
  - Consumes `response.body.getReader()` to parse tokens via `parseSSEPart`.
  - Progressively renders streamed markdown/text.
- `openFilePicker()` / `uploadFile()`:
  - Sends multipart form data to `/upload`.
  - Displays dynamic status messages ("Uploading and indexing into vector database...").
- `toggleDictation()`:
  - Interacts with browser SpeechRecognition API.
  - Automatically dictates spoken speech into the message input field.
- `handleKeyDown(e)`:
  - Pressing `Enter` sends the message.
  - Pressing `Shift + Enter` inserts a new line.

---

## 8. Configuration & Deployment Files

### 8.1 `.env` & `.env.example`
Central configuration for API credentials and operational flags:
```env
GOOGLE_API_KEY=your_key          # Required for Gemini LLM & embeddings
GOOGLE_MODEL=gemini-2.5-flash    # Default Gemini model
GEMINI_MODEL=gemini-2.5-flash    # Alternative model key
TAVILY_API_KEY=your_key          # Tavily search key
LANGSMITH_TRACING=false          # Tracing switch
LANGSMITH_PROJECT=bittugpt       # Project tag in LangSmith
```

### 8.2 `Dockerfile`
- Base: `python:3.11-slim`.
- Installs necessary build utilities (`build-essential`, `git`, `curl`).
- Installs requirements with `--no-cache-dir`.
- Creates `uploads/` and `data/` working directories.
- Exposes port `8080` and runs Uvicorn on `0.0.0.0:8080`.

### 8.3 `.github/workflows/cicd.yaml`
- Multi-job GitHub Actions pipeline:
  - `Continuous-Integration`: Runs on `ubuntu-latest`. Authenticates with AWS, builds the Docker image, and pushes to Amazon ECR.
  - `Continuous-Deployment`: Runs on a self-hosted runner on AWS EC2. Pulls the new Docker image from ECR, safely stops previous containers, and launches the updated container with configured environment secrets.
