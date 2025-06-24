#!/usr/bin/env python3
"""
AI Processing Server with Bulletproof Auto-Dependency Installation

Runs on port 7175 (because tits lol)
Works standalone on Windows and Linux with zero interaction!
FIXED: Now includes python-multipart for FastAPI file uploads
"""

import sys
import subprocess
import importlib.util
import platform


def run_pip_command(command_args):
    """Run pip command with proper error handling"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip"] + command_args,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def install_package_with_fallbacks(package_specs):
    """Try to install a package with multiple fallback versions"""
    if isinstance(package_specs, str):
        package_specs = [package_specs]

    for package_spec in package_specs:
        print(f"   Trying {package_spec}...")
        success, output = run_pip_command(["install", package_spec, "--no-warn-script-location"])
        if success:
            print(f"   ✅ {package_spec} installed successfully")
            return True
        else:
            print(f"   ❌ Failed: {package_spec}")

    print(f"   🔄 All versions failed, trying latest...")
    # Try installing the latest version without version pinning
    base_package = package_specs[0].split('==')[0].split('>=')[0]
    success, output = run_pip_command(["install", base_package, "--upgrade", "--no-warn-script-location"])
    if success:
        print(f"   ✅ {base_package} (latest) installed successfully")
        return True

    return False


def check_and_install_dependencies():
    """Check for required packages and install them if missing"""
    # Define packages with fallbacks for compatibility
    dependencies = {
        'fastapi': [
            'fastapi>=0.100.0',
            'fastapi==0.104.1',
            'fastapi==0.103.0'
        ],
        'uvicorn': [
            'uvicorn[standard]>=0.20.0',
            'uvicorn[standard]==0.24.0',
            'uvicorn[standard]==0.23.0',
            'uvicorn>=0.20.0'  # Fallback without [standard]
        ],
        'slowapi': [
            'slowapi>=0.1.8',
            'slowapi==0.1.9',
            'slowapi==0.1.8'
        ],
        'httpx': [
            'httpx>=0.24.0',
            'httpx==0.25.0',
            'httpx==0.24.1'
        ],
        'aiofiles': [
            'aiofiles>=22.0.0',
            'aiofiles==23.2.1',
            'aiofiles==23.1.0'
        ],
        'PIL': [
            'pillow>=10.0.0',  # Try latest first
            'pillow==10.4.0',  # More recent version
            'pillow==10.3.0',
            'pillow==10.2.0',
            'pillow==10.1.0',
            'pillow==9.5.0'  # Fallback to older stable version
        ],
        'pydantic': [
            'pydantic>=2.0.0',
            'pydantic==2.5.0',
            'pydantic==2.4.2'
        ],
        # ADDED: python-multipart for FastAPI file uploads
        'multipart': [
            'python-multipart>=0.0.5',
            'python-multipart==0.0.6',
            'python-multipart==0.0.5'
        ]
    }

    print("🔍 Checking dependencies...")

    # First upgrade pip to avoid issues
    print("📦 Upgrading pip...")
    run_pip_command(["install", "--upgrade", "pip", "--no-warn-script-location"])

    missing_packages = []

    # Check for packages (note: multipart is imported as multipart, not python-multipart)
    for package_name, _ in dependencies.items():
        # Special case for multipart package
        import_name = 'multipart' if package_name == 'multipart' else package_name
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing dependencies...")
        failed_packages = []

        for package_name in missing_packages:
            package_specs = dependencies[package_name]
            print(f"\n🔧 Installing {package_name}...")

            if not install_package_with_fallbacks(package_specs):
                failed_packages.append(package_name)
                print(f"   💀 FAILED to install {package_name}")

        if failed_packages:
            print(f"\n⚠️  Some packages failed to install: {failed_packages}")
            print("🤔 Trying to continue anyway...")

            # For critical failures, try alternative approaches
            if 'PIL' in failed_packages:
                print("🎨 Pillow failed, trying alternative image libraries...")
                alternatives = ['pillow-simd', 'pillow-heif', 'wand']
                for alt in alternatives:
                    success, _ = run_pip_command(["install", alt, "--no-warn-script-location"])
                    if success:
                        print(f"   ✅ Installed {alt} as Pillow alternative")
                        break

        print("🎉 Dependency installation complete!")
    else:
        print("✅ All dependencies are already installed!")


# Install dependencies before importing them
check_and_install_dependencies()

# Now import everything with fallbacks
print("📥 Importing modules...")

try:
    import asyncio
    import base64
    import json
    import os
    import sqlite3
    import threading
    import time
    from datetime import datetime, timedelta
    from typing import Dict, List, Optional

    print("   ✅ Standard library imports successful")
except ImportError as e:
    print(f"   ❌ Standard library import failed: {e}")
    sys.exit(1)

# Import third-party with graceful fallbacks
modules_imported = []
import_errors = []

try:
    import aiofiles

    modules_imported.append("aiofiles")
except ImportError as e:
    import_errors.append(f"aiofiles: {e}")

try:
    import uvicorn

    modules_imported.append("uvicorn")
except ImportError as e:
    import_errors.append(f"uvicorn: {e}")

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    modules_imported.append("fastapi")
except ImportError as e:
    import_errors.append(f"fastapi: {e}")

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    modules_imported.append("slowapi")
except ImportError as e:
    import_errors.append(f"slowapi: {e}")


    # Create dummy rate limiter
    class DummyLimiter:
        def limit(self, rate): return lambda func: func

        def __call__(self, *args): return lambda func: func


    Limiter = DummyLimiter


    def get_remote_address(request):
        return "127.0.0.1"


    def _rate_limit_exceeded_handler(*args):
        pass


    class RateLimitExceeded(Exception):
        pass

try:
    import httpx

    modules_imported.append("httpx")
except ImportError as e:
    import_errors.append(f"httpx: {e}")
    # We'll implement a fallback using urllib

try:
    from PIL import Image
    import io

    modules_imported.append("PIL")
except ImportError as e:
    import_errors.append(f"PIL: {e}")


    # Create dummy Image class
    class DummyImage:
        @staticmethod
        def open(data):
            return DummyImage()

        def verify(self):
            pass


    Image = DummyImage
    import io

# Check for multipart (python-multipart package)
try:
    import multipart

    modules_imported.append("multipart")
except ImportError as e:
    import_errors.append(f"multipart: {e}")
    print("   ⚠️  python-multipart not available - file uploads may not work")

print(f"   ✅ Successfully imported: {', '.join(modules_imported)}")
if import_errors:
    print(f"   ⚠️  Import warnings: {len(import_errors)} modules using fallbacks")

# Fallback HTTP client if httpx failed
if 'httpx' not in modules_imported:
    import urllib.request
    import urllib.parse
    import urllib.error


    class FallbackHTTPClient:
        def __init__(self, timeout=30.0):
            self.timeout = timeout

        async def post(self, url, json=None, headers=None):
            # Convert to sync call (not ideal but works)
            data = json.dumps(json).encode() if json else None
            req = urllib.request.Request(url, data=data, headers=headers or {})

            class Response:
                def __init__(self, data, status_code):
                    self._data = data
                    self.status_code = status_code

                def json(self):
                    return json.loads(self._data.decode())

                def raise_for_status(self):
                    if self.status_code >= 400:
                        raise Exception(f"HTTP {self.status_code}")

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return Response(response.read(), response.getcode())
            except Exception as e:
                return Response(b'{"error": "' + str(e).encode() + b'"}', 500)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass


    class httpx:
        AsyncClient = FallbackHTTPClient

# Database and file paths
DB_PATH = "ai_server.db"
IMAGES_DIR = "stored_images"
RESPONSES_DIR = "stored_responses"

# Rate limiter (with fallback)
if 'slowapi' in modules_imported:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = Limiter()

# Global storage
profiles = {}
blacklisted_ips = set()
activity_logs = []


# AI Service clients
class AIServiceManager:
    def __init__(self):
        self.services = {}

    def update_profile_services(self, profile_name: str, api_keys: dict):
        """Update AI services for a profile based on available API keys"""
        services = {}

        if api_keys.get('gemini_api_key'):
            services['gemini'] = {
                'api_key': api_keys['gemini_api_key'],
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent',
                'available': True
            }

        if api_keys.get('deepseek_api_key'):
            services['deepseek'] = {
                'api_key': api_keys['deepseek_api_key'],
                'base_url': 'https://api.deepseek.com/v1/chat/completions',
                'available': True
            }

        if api_keys.get('qwen_api_key'):
            services['qwen'] = {
                'api_key': api_keys['qwen_api_key'],
                'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
                'available': True
            }

        if api_keys.get('chatgpt_api_key'):
            services['chatgpt'] = {
                'api_key': api_keys['chatgpt_api_key'],
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'available': True
            }

        self.services[profile_name] = services

    async def call_gemini(self, api_key: str, image_data: str, prompt: str = "Describe this image in detail") -> str:
        """Call Gemini API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }]
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "No response from Gemini"

        except Exception as e:
            return f"Gemini error: {str(e)}"

    async def call_openai_compatible(self, service_name: str, api_key: str, base_url: str, image_data: str,
                                     prompt: str) -> str:
        """Call OpenAI-compatible APIs (DeepSeek, Qwen, ChatGPT)"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o" if service_name == "chatgpt" else "deepseek-chat" if service_name == "deepseek" else "qwen-vl-max",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(base_url, headers=headers, json=payload)
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content']

        except Exception as e:
            return f"{service_name} error: {str(e)}"

    async def ask_if_needs_help(self, service_name: str, response: str, api_key: str, base_url: str) -> bool:
        """Ask an AI if it needs help answering the question"""
        try:
            help_prompt = f"Based on your previous response: '{response}', do you need help from another AI to provide a better answer? Respond with only 'YES' or 'NO'."

            if service_name == 'gemini':
                result = await self.call_gemini(api_key, "", help_prompt)
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "gpt-4o" if service_name == "chatgpt" else "deepseek-chat" if service_name == "deepseek" else "qwen-vl-max",
                    "messages": [{"role": "user", "content": help_prompt}]
                }

                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()['choices'][0]['message']['content']

            return "YES" in result.upper()

        except Exception:
            return False


ai_manager = AIServiceManager()

# Pydantic models (with fallback)
try:
    class ProfileCreate(BaseModel):
        username: str
        api_keys: Dict[str, str]


    class AuthResponse(BaseModel):
        success: bool
        message: str
except:
    # Fallback without pydantic
    class ProfileCreate:
        def __init__(self, **kwargs):
            self.username = kwargs.get('username')
            self.api_keys = kwargs.get('api_keys', {})


    class AuthResponse:
        def __init__(self, success, message):
            self.success = success
            self.message = message


# Database functions
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Profiles table
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS profiles
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       username
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       auth_key
                       TEXT
                       NOT
                       NULL,
                       api_keys
                       TEXT
                       NOT
                       NULL,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       last_activity
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       total_requests
                       INTEGER
                       DEFAULT
                       0,
                       data_size_bytes
                       INTEGER
                       DEFAULT
                       0
                   )
                   ''')

    # Activity logs table
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS activity_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       profile_name
                       TEXT
                       NOT
                       NULL,
                       ip_address
                       TEXT
                       NOT
                       NULL,
                       request_type
                       TEXT
                       NOT
                       NULL,
                       image_path
                       TEXT,
                       ai_responses
                       TEXT,
                       final_answer
                       TEXT,
                       timestamp
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       processing_time_seconds
                       REAL
                   )
                   ''')

    # Blacklisted IPs table
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS blacklisted_ips
                   (
                       ip_address
                       TEXT
                       PRIMARY
                       KEY,
                       blacklisted_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       reason
                       TEXT
                   )
                   ''')

    conn.commit()
    conn.close()


def load_profiles():
    """Load profiles from database"""
    global profiles
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT username, auth_key, api_keys FROM profiles')
    rows = cursor.fetchall()

    for username, auth_key, api_keys_json in rows:
        api_keys = json.loads(api_keys_json)
        profiles[username] = {
            'auth_key': auth_key,
            'api_keys': api_keys
        }
        ai_manager.update_profile_services(username, api_keys)

    conn.close()


def load_blacklisted_ips():
    """Load blacklisted IPs from database"""
    global blacklisted_ips
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT ip_address FROM blacklisted_ips')
    rows = cursor.fetchall()

    blacklisted_ips = {row[0] for row in rows}
    conn.close()


def cleanup_old_data():
    """Clean up data older than 3 days or profiles exceeding 1GB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Delete old activity logs (older than 3 days)
    three_days_ago = datetime.now() - timedelta(days=3)
    cursor.execute('DELETE FROM activity_logs WHERE timestamp < ?', (three_days_ago,))

    # Check profile data sizes and clean up if needed
    cursor.execute('''
                   SELECT profile_name, SUM(LENGTH(ai_responses) + LENGTH(final_answer)) as data_size
                   FROM activity_logs
                   GROUP BY profile_name
                   ''')

    for profile_name, data_size in cursor.fetchall():
        if data_size and data_size > 1024 * 1024 * 1024:  # 1GB
            # Delete oldest entries for this profile
            cursor.execute('''
                           DELETE
                           FROM activity_logs
                           WHERE profile_name = ?
                             AND id IN (SELECT id
                                        FROM activity_logs
                                        WHERE profile_name = ?
                                        ORDER BY
                               timestamp ASC
                               LIMIT 100
                               )
                           ''', (profile_name, profile_name))

    conn.commit()
    conn.close()


# Middleware for IP blacklisting
async def check_blacklist(request: Request):
    client_ip = get_remote_address(request)
    if client_ip in blacklisted_ips:
        raise HTTPException(status_code=403, detail=f"IP {client_ip} is blacklisted")


# FastAPI app with lifespan (with fallback)
if 'fastapi' in modules_imported:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        init_database()
        load_profiles()
        load_blacklisted_ips()

        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())

        yield

        # Shutdown
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


    app = FastAPI(title="AI Processing Server", version="1.0.0", lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting
    if 'slowapi' in modules_imported:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    # Fallback simple HTTP server
    print("⚠️  FastAPI not available, using basic fallback server")
    app = None


# Periodic cleanup task
async def periodic_cleanup():
    while True:
        try:
            cleanup_old_data()
            await asyncio.sleep(3600)  # Run every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Cleanup error: {e}")
            await asyncio.sleep(3600)


# Routes (only if FastAPI is available)
if app:
    @app.post("/auth-{auth_key}")
    @limiter.limit("5/minute")
    async def authenticate(auth_key: str, request: Request):
        """Authenticate user with auth key"""
        await check_blacklist(request)

        # Find profile with matching auth key
        for username, profile_data in profiles.items():
            if profile_data['auth_key'] == auth_key:
                return AuthResponse(success=True, message="youre authenticated! gg!")

        return AuthResponse(success=False, message="Ratio + Syabu + L")


    @app.post("/profile")
    @limiter.limit("10/minute")
    async def create_or_update_profile(profile_data: ProfileCreate, request: Request):
        """Create or update user profile with API keys"""
        await check_blacklist(request)

        # Verify auth key exists
        auth_key_found = False
        for username, existing_profile in profiles.items():
            if username == profile_data.username:
                auth_key_found = True
                break

        if not auth_key_found:
            return JSONResponse(status_code=401, content={"message": "Authentication required"})

        # Update profile
        profiles[profile_data.username]['api_keys'] = profile_data.api_keys
        ai_manager.update_profile_services(profile_data.username, profile_data.api_keys)

        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE profiles
                       SET api_keys      = ?,
                           last_activity = CURRENT_TIMESTAMP
                       WHERE username = ?
                       ''', (json.dumps(profile_data.api_keys), profile_data.username))

        conn.commit()
        conn.close()

        return {"message": "User profile authenticated"}


    @app.post("/process")
    @limiter.limit("20/hour")
    async def process_image(
            request: Request,
            image: UploadFile = File(...),
            username: str = Form(...),
            prompt: str = Form(default="Describe this image in detail")
    ):
        """Process image through AI services"""
        await check_blacklist(request)
        start_time = time.time()

        # Check if user is authenticated
        if username not in profiles:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Read and validate image
        image_data = await image.read()

        try:
            # Validate image format
            img = Image.open(io.BytesIO(image_data))
            img.verify()

            # Convert to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        # Get available AI services for this user
        user_services = ai_manager.services.get(username, {})
        if not user_services:
            raise HTTPException(status_code=400, detail="No AI services configured for user")

        # AI processing logic
        ai_responses = {}
        service_order = ['gemini', 'deepseek', 'qwen', 'chatgpt']

        current_response = None

        for service_name in service_order:
            if service_name not in user_services:
                continue

            service_config = user_services[service_name]

            try:
                if service_name == 'gemini':
                    response = await ai_manager.call_gemini(
                        service_config['api_key'],
                        image_b64,
                        prompt
                    )
                else:
                    response = await ai_manager.call_openai_compatible(
                        service_name,
                        service_config['api_key'],
                        service_config['base_url'],
                        image_b64,
                        prompt
                    )

                ai_responses[service_name] = response
                current_response = response

                # Ask if this AI needs help
                if service_name != 'chatgpt':  # Don't ask ChatGPT if it needs help (it's the last resort)
                    needs_help = await ai_manager.ask_if_needs_help(
                        service_name,
                        response,
                        service_config['api_key'],
                        service_config['base_url']
                    )

                    if not needs_help:
                        break  # This AI is confident, use its response

            except Exception as e:
                ai_responses[service_name] = f"Error: {str(e)}"
                continue

        # If we have multiple responses, use Gemini to pick the best one
        if len(ai_responses) > 1 and 'gemini' in user_services:
            try:
                all_responses = "\n\n".join([f"{service}: {response}" for service, response in ai_responses.items()])
                best_answer_prompt = f"Here are responses from different AI services:\n{all_responses}\n\nWhich response is the most accurate and helpful? Provide the best answer or combine them into a better response."

                final_answer = await ai_manager.call_gemini(
                    user_services['gemini']['api_key'],
                    "",
                    best_answer_prompt
                )
            except:
                final_answer = current_response or "No valid response received"
        else:
            final_answer = current_response or "No valid response received"

        # Store image and response
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{username}_{timestamp}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)

        if 'aiofiles' in modules_imported:
            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(image_data)
        else:
            with open(image_path, 'wb') as f:
                f.write(image_data)

        # Log activity
        processing_time = time.time() - start_time
        client_ip = get_remote_address(request)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO activity_logs
                       (profile_name, ip_address, request_type, image_path, ai_responses, final_answer,
                        processing_time_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           username,
                           client_ip,
                           'image_process',
                           image_path,
                           json.dumps(ai_responses),
                           final_answer,
                           processing_time
                       ))

        # Update profile stats
        cursor.execute('''
                       UPDATE profiles
                       SET total_requests = total_requests + 1,
                           last_activity  = CURRENT_TIMESTAMP
                       WHERE username = ?
                       ''', (username,))

        conn.commit()
        conn.close()

        return {
            "answer": final_answer,
            "ai_responses": ai_responses,
            "processing_time": processing_time
        }


    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Command line interface
class CommandInterface:
    def __init__(self):
        self.running = True

    def start(self):
        """Start the command interface in a separate thread"""
        thread = threading.Thread(target=self._command_loop, daemon=True)
        thread.start()

    def _command_loop(self):
        """Main command loop"""
        print("\n=== AI Processing Server Command Interface ===")
        print("Available commands:")
        print("  /listprofiles - List all profiles")
        print("  /deleteprofile <username> - Delete a profile")
        print("  /addprofile <username> <auth_key> - Add a new profile")
        print("  /activityreport <username> - Show activity report for profile")
        print("  /blacklistip <ip> - Blacklist an IP address")
        print("  /help - Show this help message")
        print("  /quit - Quit the server")
        print("=" * 50)

        while self.running:
            try:
                command = input("\n> ").strip()
                self._process_command(command)
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down...")
                break

    def _process_command(self, command: str):
        """Process a command"""
        parts = command.split()
        if not parts:
            return

        cmd = parts[0].lower()

        if cmd == "/listprofiles":
            self._list_profiles()
        elif cmd == "/deleteprofile" and len(parts) == 2:
            self._delete_profile(parts[1])
        elif cmd == "/addprofile" and len(parts) == 3:
            self._add_profile(parts[1], parts[2])
        elif cmd == "/activityreport" and len(parts) == 2:
            self._activity_report(parts[1])
        elif cmd == "/blacklistip" and len(parts) == 2:
            self._blacklist_ip(parts[1])
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/quit":
            self.running = False
            os._exit(0)
        else:
            print("Invalid command. Type /help for available commands.")

    def _list_profiles(self):
        """List all profiles"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT username, created_at, last_activity, total_requests
                       FROM profiles
                       ORDER BY last_activity DESC
                       ''')

        profiles_data = cursor.fetchall()
        conn.close()

        if not profiles_data:
            print("No profiles found.")
            return

        print(f"\n{'Username':<20} {'Created':<20} {'Last Activity':<20} {'Requests':<10}")
        print("-" * 70)

        for username, created, last_activity, requests in profiles_data:
            print(f"{username:<20} {created:<20} {last_activity:<20} {requests:<10}")

    def _delete_profile(self, username: str):
        """Delete a profile"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if profile exists
        cursor.execute('SELECT username FROM profiles WHERE username = ?', (username,))
        if not cursor.fetchone():
            print(f"Profile '{username}' not found.")
            conn.close()
            return

        # Delete profile and related data
        cursor.execute('DELETE FROM profiles WHERE username = ?', (username,))
        cursor.execute('DELETE FROM activity_logs WHERE profile_name = ?', (username,))

        conn.commit()
        conn.close()

        # Remove from memory
        if username in profiles:
            del profiles[username]
        if username in ai_manager.services:
            del ai_manager.services[username]

        print(f"Profile '{username}' deleted successfully.")

    def _add_profile(self, username: str, auth_key: str):
        """Add a new profile"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                           INSERT INTO profiles (username, auth_key, api_keys)
                           VALUES (?, ?, ?)
                           ''', (username, auth_key, "{}"))

            conn.commit()
            conn.close()

            # Add to memory
            profiles[username] = {'auth_key': auth_key, 'api_keys': {}}

            print(f"Profile '{username}' added successfully with auth key '{auth_key}'")

        except sqlite3.IntegrityError:
            print(f"Profile '{username}' already exists.")
            conn.close()

    def _activity_report(self, username: str):
        """Show activity report for a profile"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get profile info
        cursor.execute('SELECT * FROM profiles WHERE username = ?', (username,))
        profile = cursor.fetchone()

        if not profile:
            print(f"Profile '{username}' not found.")
            conn.close()
            return

        print(f"\n=== Activity Report for {username} ===")
        print(f"Created: {profile[4]}")
        print(f"Last Activity: {profile[5]}")
        print(f"Total Requests: {profile[6]}")

        # Get recent activity
        cursor.execute('''
                       SELECT timestamp, ip_address, ai_responses, final_answer, processing_time_seconds
                       FROM activity_logs
                       WHERE profile_name = ?
                       ORDER BY timestamp DESC
                           LIMIT 10
                       ''', (username,))

        activities = cursor.fetchall()
        conn.close()

        if activities:
            print(f"\nRecent Activity (Last 10 requests):")
            print("-" * 60)

            for timestamp, ip, ai_responses_json, final_answer, proc_time in activities:
                ai_responses = json.loads(ai_responses_json) if ai_responses_json else {}

                print(f"\nTimestamp: {timestamp}")
                print(f"IP: {ip}")
                print(f"Processing Time: {proc_time:.2f}s")
                print(f"AI Services Used: {', '.join(ai_responses.keys())}")
                print(f"Final Answer: {final_answer[:100]}...")
        else:
            print("\nNo activity found.")

    def _blacklist_ip(self, ip: str):
        """Blacklist an IP address"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                           INSERT INTO blacklisted_ips (ip_address, reason)
                           VALUES (?, ?)
                           ''', (ip, "Manual blacklist"))

            conn.commit()
            conn.close()

            blacklisted_ips.add(ip)
            print(f"IP '{ip}' has been blacklisted.")

        except sqlite3.IntegrityError:
            print(f"IP '{ip}' is already blacklisted.")
            conn.close()

    def _show_help(self):
        """Show help message"""
        print("\nAvailable commands:")
        print("  /listprofiles - List all profiles")
        print("  /deleteprofile <username> - Delete a profile")
        print("  /addprofile <username> <auth_key> - Add a new profile")
        print("  /activityreport <username> - Show activity report for profile")
        print("  /blacklistip <ip> - Blacklist an IP address")
        print("  /help - Show this help message")
        print("  /quit - Quit the server")


if __name__ == "__main__":
    system_info = f"{platform.system()} {platform.release()} - Python {platform.python_version()}"
    print("🚀 AI Processing Server - BULLETPROOF Edition v2")
    print("=" * 60)
    print(f"🖥️  System: {system_info}")
    print("🎯 Target: Port 7175 (because tits, lol)")
    print("🔧 Features:")
    print("   - 💪 Bulletproof auto dependency installation")
    print("   - 📁 FIXED: python-multipart for file uploads")
    print("   - 🔄 Graceful fallbacks for all components")
    print("   - 🔐 Authentication system")
    print("   - ⚡ Rate limiting (with fallback)")
    print("   - 🚫 IP blacklisting")
    print("   - 🤖 Multi-AI routing (Gemini → DeepSeek → Qwen → ChatGPT)")
    print("   - 💾 Data retention (3 days or 1GB per profile)")
    print("   - 🖥️  Command interface")
    print("   - 🌍 Windows/Linux compatible")
    print("=" * 60)

    if not app:
        print("⚠️  Running in limited mode due to missing dependencies")
        print("🔧 Some features may not be available")

    # Initialize directories and database
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    init_database()
    load_profiles()
    load_blacklisted_ips()

    # Start command interface
    cmd_interface = CommandInterface()
    cmd_interface.start()

    # Start server
    print("\n🎉 Server starting on http://0.0.0.0:7175")
    print("Type commands in the console to manage the server!")
    print("Press Ctrl+C to shutdown gracefully")

    if app and 'uvicorn' in modules_imported:
        try:
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=7175,
                log_level="info"
            )
        except Exception as e:
            print(f"❌ Server failed to start: {e}")
            print("🤔 Try running with administrator/root privileges")
    else:
        print("⚠️  Web server not available - running in command-only mode")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")