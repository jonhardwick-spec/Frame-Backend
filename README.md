# 🤖 AI Processing Server - Bulletproof Edition

A robust, self-installing AI image processing server that intelligently routes requests through multiple AI providers with automatic fallback and best-answer selection, optimized for high-speed image streaming.

## 🎯 **What It Does**

This server processes images through multiple AI services in sequence, intelligently asking each AI if it needs help before deciding whether to move to the next. It automatically selects the best response, leveraging a "judge" AI if multiple responses are gathered, and returns it to your application.

## **Upcoming features**
- GPT4FREE integration!!! (Thats right boys, we are NOT getting rate limited L0L)

**AI Service Routing Logic:**
1.  **Gemini** → Fast, good for general questions, math, directions.
    * *Confidence-based routing:* If Gemini needs help, it passes to the next AI.
2.  **DeepSeek** → Smart but slower, good for complex reasoning.
    * *Confidence-based routing:* If DeepSeek needs help, it passes to the next AI.
3.  **Qwen** → Translation and language tasks.
    * *Confidence-based routing:* If Qwen needs help, it passes to the next AI.
4.  **ChatGPT** → Final fallback option.

If multiple AIs respond, Gemini (if available) acts as a "judge" to select the most accurate and helpful answer or combine them into a better response.

## ✨ **Key Features**

### 🔧 **Zero-Configuration Setup**
-   **Auto-installs all dependencies** with intelligent fallbacks (e.g., trying older `pillow` versions or `urllib` for `httpx`).
-   **Works out-of-the-box** on Windows and Linux.
-   **Graceful degradation** when packages fail to install.
-   **No manual configuration required**.

### 🔐 **Enterprise-Grade Security**
-   **Custom authentication endpoints** (`/auth-<your-key>`).
-   **IP blacklisting** with automatic persistence.
-   **Rate limiting** (5 auth/min, 10 profile updates/min, **3600 image processing/hour for 1 FPS streaming!**).
-   **Request logging** and activity monitoring.

### 🤖 **Intelligent AI Routing**
-   **Multi-provider support** (Gemini, DeepSeek, Qwen, ChatGPT).
-   **Confidence-based routing** (AIs self-assess if they need help answering).
-   **Best-answer selection** using Gemini as judge.
-   **Automatic failover** when services are unavailable.
-   **Intelligent rate limiting** with randomized jitter to prevent API overuse.

### 💾 **Smart Data Management**
-   **Automatic cleanup** (data older than 3 days or profiles exceeding 1GB).
-   **SQLite database** for profiles, activity logs, and custom prompts.
-   **Image storage** with organized file structure.
-   **Usage analytics** and reporting.

### 📱 **Flutter App Integration**
-   **NEW:** Real-time **debug messages** included in responses, cleaned of emojis for glasses display.
-   **NEW:** Flutter app can now **set and retrieve custom prompts** remotely via dedicated API endpoints.

### 🖥️ **Command-Line Management**
-   **Live server management** while running.
-   **Profile management** (`/addprofile`, `/deleteprofile`, `/listprofiles`).
-   **Activity reports** with detailed analytics.
-   **IP blacklisting** (`/blacklistip`).
-   **NEW:** **Custom prompt management** (`/prompt <username> [new_prompt]`).

## 🚀 **Quick Start**

### Prerequisites
-   Python 3.7+
-   Internet connection (for dependency installation)

### Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/jonhardwick-spec/Frame-Backend.git](https://github.com/jonhardwick-spec/Frame-Backend.git)
    cd Frame-Backend
    ```

    **Or download directly**:
    ```bash
    wget [https://raw.githubusercontent.com/jonhardwick-spec/Frame-Backend/main/main.py](https://raw.githubusercontent.com/jonhardwick-spec/Frame-Backend/main/main.py) -O server.py
    # or
    curl -O [https://raw.githubusercontent.com/jonhardwick-spec/Frame-Backend/main/main.py](https://raw.githubusercontent.com/jonhardwick-spec/Frame-Backend/main/main.py) -o server.py
    ```
    *Note: The main file is now `main.py` in the repo, but renamed to `server.py` for download convenience.*

2.  **Run it** (that's it!):
    ```bash
    python server.py
    ```

3.  **The server will**:
    -   Auto-install all required dependencies.
    -   Start on `http://0.0.0.0:7175`.
    -   Initialize the database.
    -   Launch the command interface.

### First Setup

1.  **Add a profile** via command interface:
    ```
    > /addprofile myusername mysecretkey123
    ```

2.  **Test authentication**:
    ```bash
    curl -X POST "http://localhost:7175/auth-mysecretkey123"
    ```

3.  **Configure your app** with the API keys through your Flutter app settings.

## 📡 **API Endpoints**

### Authentication
```http
POST /auth-{auth_key}
````

Authenticate using your profile's auth key.

### Profile Management

```http
POST /profile
Content-Type: application/json

{
  "username": "myusername",
  "api_keys": {
    "gemini_api_key": "AIza...",
    "deepseek_api_key": "sk-...",
    "qwen_api_key": "sk-...",
    "chatgpt_api_key": "sk-..."
  }
}
```

Update a user's API keys. The user must already exist in the database (added via command interface).

### Prompt Management (NEW\!)

Allows Flutter apps to set and retrieve custom prompts.

**Set Custom Prompt:**

```http
POST /prompt-{username}
Content-Type: application/json

{
  "prompt": "Analyze this image for potential security threats and vulnerabilities."
}
```

Set a custom prompt for a specific user. If the prompt is empty, it reverts to the default.

**Get Current Prompt:**

```http
GET /prompt-{username}
```

Retrieve the current custom prompt for a user, or the default prompt if no custom prompt is set.

**Response:**

```json
{
  "username": "myusername",
  "prompt": "Analyze this image for potential security threats and vulnerabilities.",
  "is_custom": true,
  "default_prompt": "Describe this image in detail"
}
```

### Image Processing

```http
POST /process
Content-Type: multipart/form-data

image: [image file]
username: myusername
prompt: "Describe this image" (optional, will use user's custom prompt or default if not provided)
```

**Response:**

```json
{
  "answer": "The best selected response",
  "ai_responses": {
    "gemini": "Gemini's response",
    "deepseek": "DeepSeek's response"
  },
  "processing_time": 2.34,
  "prompt_used": "Describe this image",
  "debug_messages": [
    "[10:30:05] [AI] Processing image for myusername",
    "[10:30:05] [TARGET] Using prompt: 'Describe this image...' "
  ],
  "success": true,
  "error": null
}
```

Includes `debug_messages` (emoji-free) for direct display on connected devices (e.g., smart glasses).

### Health Check

```http
GET /health
```

## 🖥️ **Command Interface**

While the server is running, use these commands in the console:

| Command | Description |
|-------------------------------------|-----------------------------------------------------------------------------|
| `/listprofiles` | List all user profiles (📝=custom prompt, 📄=default). |
| `/addprofile <username> <auth_key>` | Add a new user profile to the system. |
| `/deleteprofile <username>` | Delete a profile and all associated data (images, logs, custom prompts). |
| `/activityreport <username>` | Show detailed usage statistics and recent activity for a profile. |
| `/blacklistip <ip>` | Block an IP address from accessing the server. |
| `/prompt <username>` | View the current custom or default prompt for a user. |
| `/prompt <username> [new_prompt]` | Set a custom prompt for a user. Example: `/prompt john [Analyze for security]` |
| `/help` | Show all available commands and usage instructions. |
| `/quit` | Gracefully shutdown the server. |

## 🛠️ **Configuration**

### Supported AI Providers

| Provider | Image Support | Notes |
|----------|---------------|-------|
| **Google Gemini** | ✅ | Fast, good for general tasks. |
| **DeepSeek** | ✅ | Slower but very capable. |
| **Qwen** | ✅ | Great for languages/translation. |
| **ChatGPT (GPT-4o)** | ✅ | Reliable fallback option. |

### Rate Limits

  - **Authentication**: 5 requests/minute
  - **Profile Updates**: 10 requests/minute
  - **Image Processing**: **3600 requests/hour** (Designed for 1 Frame Per Second streaming\!)

### Data Retention

  - **Images & Responses**: Cleaned up after 3 days or if a profile's data exceeds 1GB.
  - **Activity Logs**: Automatically cleaned up (older than 3 days).
  - **Database**: SQLite with automatic initialization and maintenance.

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flutter App   │───▶│   Auth Gateway  │───▶│    AI Router    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                   │                              │
                                   ▼                              ▼
                 ┌─────────────────┐          ┌─────────────────┐
                 │   Rate Limiter  │          │ AI Services     │
                 └─────────────────┘          │ ┌─────────────┐ │
                                   │          │ │   Gemini    │ │
                 ┌─────────────────┐          │ │  DeepSeek   │ │
                 │    Database     │          │ │    Qwen     │ │
                 │   (SQLite)      │          │ │   ChatGPT   │ │
                 └─────────────────┘          │ └─────────────┘ │
                                   │          └─────────────────┘
                 ┌─────────────────┐                │
                 │  File Storage   │                ▼
                 │ (Images/Logs)   │          ┌─────────────────┐
                 └─────────────────┘          │ Response Judge  │
                                              │   (Gemini)      │
                                              └─────────────────┘
```

## 🔧 **Development**

### Dependencies (Auto-installed)

  - `fastapi>=0.100.0` - Web framework.
  - `uvicorn[standard]>=0.20.0` - ASGI server.
  - `httpx>=0.24.0` - HTTP client for AI APIs (with `urllib` fallback).
  - `slowapi>=0.1.8` - Rate limiting (with dummy fallback).
  - `pillow>=10.0.0` - Image processing (with fallbacks to older versions/alternatives).
  - `aiofiles>=22.0.0` - Async file operations (with sync fallback).
  - `python-multipart>=0.0.5` - File upload support (with warning if missing).
  - `pydantic>=2.0.0` - Data validation (with basic class fallback).

### Project Structure

```
├── main.py                # Main server file (self-contained)
├── ai_server.db           # SQLite database (auto-created)
├── stored_images/         # Image storage (auto-created)
└── stored_responses/      # Response logs (auto-created)
```

## 🤝 **Contributing**

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jonhardwick-spec/Frame-Backend/blob/main/LICENSE) file for details.

## 🎉 **Why Port 7175?**

Because it spells "TITS" upside down on a calculator, and we're not too mature for that. 😄

-----

**Made with ❤️ for developers who want AI image processing without the hassle.**

```
```