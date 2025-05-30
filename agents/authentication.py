"""
Authentication agent for the web scraping system.
"""
import os
import asyncio
import logging
import time
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import httpx
from urllib.parse import urlparse
from datetime import datetime, timedelta
import re
import aiohttp
from playwright.async_api import async_playwright
import jwt
from oauthlib.oauth2 import WebApplicationClient
import requests_oauthlib
import secrets
import uuid
from requests_oauthlib import OAuth1Session
from Crypto.Cipher import AES

from agents.base import Agent
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage


class CredentialStore:
    """
    Secure storage for website credentials.
    """
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize a new credential store.

        Args:
            storage_path: Path to store encrypted credentials. If None, credentials are stored in memory only.
        """
        self.storage_path = storage_path
        self.credentials = {}
        self.logger = logging.getLogger("credential_store")

        # Load credentials if storage path is provided
        if storage_path:
            self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from storage."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                encrypted_data = f.read()

            # In a real implementation, this would decrypt the data
            # For now, we'll just use a simple encoding
            data = base64.b64decode(encrypted_data).decode("utf-8")
            self.credentials = json.loads(data)

            self.logger.info(f"Loaded credentials for {len(self.credentials)} domains")
        except Exception as e:
            self.logger.error(f"Error loading credentials: {str(e)}")

    def _save_credentials(self) -> None:
        """Save credentials to storage."""
        if not self.storage_path:
            return

        try:
            # In a real implementation, this would encrypt the data
            # For now, we'll just use a simple encoding
            data = json.dumps(self.credentials)
            encrypted_data = base64.b64encode(data.encode("utf-8")).decode("utf-8")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)

            with open(self.storage_path, "w") as f:
                f.write(encrypted_data)

            self.logger.info(f"Saved credentials for {len(self.credentials)} domains")
        except Exception as e:
            self.logger.error(f"Error saving credentials: {str(e)}")

    def get_credentials(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get credentials for a domain.

        Args:
            domain: The domain to get credentials for.

        Returns:
            A dictionary containing the credentials, or None if not found.
        """
        return self.credentials.get(domain)

    def store_credentials(self, domain: str, credentials: Dict[str, Any]) -> None:
        """
        Store credentials for a domain.

        Args:
            domain: The domain to store credentials for.
            credentials: The credentials to store.
        """
        self.credentials[domain] = credentials
        self._save_credentials()

    def remove_credentials(self, domain: str) -> None:
        """
        Remove credentials for a domain.

        Args:
            domain: The domain to remove credentials for.
        """
        if domain in self.credentials:
            del self.credentials[domain]
            self._save_credentials()


class CaptchaSolver:
    """
    Solver for CAPTCHA challenges.
    """
    def __init__(self, api_key: Optional[str] = None, service: str = "2captcha"):
        """
        Initialize a new CAPTCHA solver.

        Args:
            api_key: API key for the CAPTCHA solving service.
            service: CAPTCHA solving service to use.
        """
        self.api_key = api_key
        self.service = service
        self.logger = logging.getLogger("captcha_solver")

    async def solve_recaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve a reCAPTCHA challenge.

        Args:
            site_key: The site key for the reCAPTCHA.
            page_url: The URL of the page containing the reCAPTCHA.

        Returns:
            The solution token, or None if solving failed.
        """
        if not self.api_key:
            self.logger.warning("No API key provided for CAPTCHA solving")
            return None

        self.logger.info(f"Solving reCAPTCHA for {page_url}")

        # This is a placeholder for actual CAPTCHA solving
        # In a real implementation, you would use a CAPTCHA solving service API

        # Simulate CAPTCHA solving with a delay
        await asyncio.sleep(5.0)

        # Return a fake token
        return "03AGdBq24PBCbwiDRgC3...fake_token"

    async def solve_hcaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve an hCaptcha challenge.

        Args:
            site_key: The site key for the hCaptcha.
            page_url: The URL of the page containing the hCaptcha.

        Returns:
            The solution token, or None if solving failed.
        """
        if not self.api_key:
            self.logger.warning("No API key provided for CAPTCHA solving")
            return None

        self.logger.info(f"Solving hCaptcha for {page_url}")

        # This is a placeholder for actual CAPTCHA solving
        # In a real implementation, you would use a CAPTCHA solving service API

        # Simulate CAPTCHA solving with a delay
        await asyncio.sleep(5.0)

        # Return a fake token
        return "P0_eyJ0eXAiOiJKV...fake_token"

    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """
        Solve an image CAPTCHA.

        Args:
            image_data: The image data as bytes.

        Returns:
            The solution text, or None if solving failed.
        """
        if not self.api_key:
            self.logger.warning("No API key provided for CAPTCHA solving")
            return None

        self.logger.info("Solving image CAPTCHA")

        # This is a placeholder for actual CAPTCHA solving
        # In a real implementation, you would use a CAPTCHA solving service API

        # Simulate CAPTCHA solving with a delay
        await asyncio.sleep(2.0)

        # Return a fake solution
        return "AB123"


class AuthenticationAgent(Agent):
    """
    Agent responsible for handling various authentication methods.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new authentication agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="authentication")
        self.coordinator_id = coordinator_id

        # Initialize browser for JavaScript-based authentication
        self.browser = None
        self.context = None

        # Enhanced session storage with encryption
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl = 3600  # 1 hour
        self.session_encryption_key = os.getenv("SESSION_ENCRYPTION_KEY", secrets.token_hex(32))

        # OAuth clients
        self.oauth1_clients: Dict[str, OAuth1Session] = {}
        self.oauth2_clients: Dict[str, WebApplicationClient] = {}

        # SAML configuration
        self.saml_config: Dict[str, Any] = {}
        self.saml_certificates: Dict[str, str] = {}

        # JWT configuration
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")
        self.jwt_algorithm = "HS256"

        # API key storage with enhanced security
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.api_key_encryption_key = os.getenv("API_KEY_ENCRYPTION_KEY", secrets.token_hex(32))

        # Register message handlers
        self.register_handler("authenticate", self._handle_authenticate)
        self.register_handler("refresh_token", self._handle_refresh_token)
        self.register_handler("validate_session", self._handle_validate_session)
        self.register_handler("revoke_session", self._handle_revoke_session)
        self.register_handler("get_session_info", self._handle_get_session_info)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the authentication agent."""
        asyncio.create_task(self._periodic_session_cleanup())
        asyncio.create_task(self._periodic_token_refresh())

    async def _periodic_session_cleanup(self) -> None:
        """Periodically clean up expired sessions."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if not self.running:
                    break

                current_time = time.time()
                expired_sessions = [
                    session_id for session_id, session in self.sessions.items()
                    if current_time - session["created_at"] > self.session_ttl
                ]

                for session_id in expired_sessions:
                    await self._revoke_session(session_id)

            except Exception as e:
                self.logger.error(f"Error in session cleanup: {str(e)}", exc_info=True)

    async def _periodic_token_refresh(self) -> None:
        """Periodically refresh OAuth2 tokens."""
        while self.running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                if not self.running:
                    break

                for session_id, session in self.sessions.items():
                    if session["auth_type"] == "oauth2" and "refresh_token" in session:
                        try:
                            await self._refresh_oauth2_token(session_id)
                        except Exception as e:
                            self.logger.error(f"Error refreshing token for session {session_id}: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error in token refresh: {str(e)}", exc_info=True)

    async def _handle_authenticate(self, message: Message) -> None:
        """Handle authentication request."""
        try:
            auth_type = message.data.get("auth_type")
            credentials = message.data.get("credentials", {})

            if auth_type == "basic":
                result = await self._authenticate_basic(credentials)
            elif auth_type == "form":
                result = await self._authenticate_form(credentials)
            elif auth_type == "oauth1":
                result = await self._authenticate_oauth1(credentials)
            elif auth_type == "oauth2":
                result = await self._authenticate_oauth2(credentials)
            elif auth_type == "saml":
                result = await self._authenticate_saml(credentials)
            elif auth_type == "jwt":
                result = await self._authenticate_jwt(credentials)
            elif auth_type == "api_key":
                result = await self._authenticate_api_key(credentials)
            else:
                raise ValueError(f"Unsupported authentication type: {auth_type}")

            # Create encrypted session
            session_id = str(uuid.uuid4())
            encrypted_session = self._encrypt_session_data({
                "auth_type": auth_type,
                "credentials": result,
                "created_at": time.time(),
                "last_activity": time.time(),
                "ip_address": message.data.get("ip_address"),
                "user_agent": message.data.get("user_agent")
            })

            self.sessions[session_id] = encrypted_session

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"session_id": session_id, "credentials": result}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="authentication_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _authenticate_basic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate using Basic Authentication.

        Args:
            parameters: Authentication parameters including username and password.

        Returns:
            A dictionary containing authentication results.
        """
        username = parameters.get("username")
        password = parameters.get("password")

        if not username or not password:
            raise ValueError("Missing username or password")

        # Create Basic Auth header
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        # Store session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "type": "basic",
            "headers": headers,
            "timestamp": time.time()
        }

        return {
            "session_id": session_id,
            "headers": headers
        }

    async def _authenticate_form(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate using form-based authentication.

        Args:
            parameters: Authentication parameters including form details.

        Returns:
            A dictionary containing authentication results.
        """
        if not self.context:
            raise RuntimeError("Browser context not initialized")

        url = parameters.get("url")
        username = parameters.get("username")
        password = parameters.get("password")
        username_field = parameters.get("username_field", "username")
        password_field = parameters.get("password_field", "password")
        submit_button = parameters.get("submit_button")

        if not all([url, username, password]):
            raise ValueError("Missing required form authentication parameters")

        page = await self.context.new_page()
        try:
            # Navigate to login page
            await page.goto(url)

            # Fill in form
            await page.fill(f"input[name='{username_field}']", username)
            await page.fill(f"input[name='{password_field}']", password)

            # Submit form
            if submit_button:
                await page.click(submit_button)
            else:
                await page.keyboard.press("Enter")

            # Wait for navigation
            await page.wait_for_load_state("networkidle")

            # Get cookies
            cookies = await page.context.cookies()

            # Store session
            session_id = secrets.token_urlsafe(32)
            self.sessions[session_id] = {
                "type": "form",
                "cookies": cookies,
                "timestamp": time.time()
            }

            return {
                "session_id": session_id,
                "cookies": cookies
            }

        finally:
            await page.close()

    async def _authenticate_oauth1(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using OAuth1."""
        consumer_key = credentials.get("consumer_key")
        consumer_secret = credentials.get("consumer_secret")
        access_token = credentials.get("access_token")
        access_token_secret = credentials.get("access_token_secret")

        if not all([consumer_key, consumer_secret]):
            raise ValueError("Missing required OAuth1 credentials")

        # Create OAuth1 client
        client = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret
        )
        self.oauth1_clients[consumer_key] = client

        # Verify credentials
        try:
            response = client.get(credentials.get("verify_url", "https://api.twitter.com/1.1/account/verify_credentials.json"))
            response.raise_for_status()

            return {
                "consumer_key": consumer_key,
                "access_token": access_token,
                "access_token_secret": access_token_secret,
                "verified": True
            }
        except Exception as e:
            raise ValueError(f"OAuth1 verification failed: {str(e)}")

    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using OAuth2."""
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        token_url = credentials.get("token_url")
        scope = credentials.get("scope", [])

        if not all([client_id, client_secret, token_url]):
            raise ValueError("Missing required OAuth2 credentials")

        # Create OAuth2 client
        client = WebApplicationClient(client_id)
        self.oauth2_clients[client_id] = client

        # Get token
        token = await client.get_token(scope=scope)

        return {
            "access_token": token["access_token"],
            "refresh_token": token.get("refresh_token"),
            "expires_in": token.get("expires_in"),
            "token_type": token.get("token_type", "Bearer")
        }

    async def _authenticate_saml(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using SAML."""
        entity_id = credentials.get("entity_id")
        assertion_consumer_service_url = credentials.get("assertion_consumer_service_url")
        idp_url = credentials.get("idp_url")

        if not all([entity_id, assertion_consumer_service_url, idp_url]):
            raise ValueError("Missing required SAML credentials")

        # Load SAML configuration
        if entity_id not in self.saml_config:
            self.saml_config[entity_id] = {
                "entity_id": entity_id,
                "assertion_consumer_service_url": assertion_consumer_service_url,
                "idp_url": idp_url,
                "certificate": credentials.get("certificate")
            }

        # Create SAML request
        auth_request = self._create_saml_auth_request(entity_id)

        # Send request to IdP
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(idp_url, data=auth_request) as response:
                    if response.status != 200:
                        raise ValueError(f"SAML authentication failed: {response.status}")

                    # Parse SAML response
                    saml_response = await response.text()
                    assertion = self._parse_saml_response(saml_response)

                    return {
                        "entity_id": entity_id,
                        "assertion": assertion,
                        "expires_at": time.time() + 3600  # 1 hour
                    }
        except Exception as e:
            raise ValueError(f"SAML authentication failed: {str(e)}")

    async def _authenticate_jwt(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using JWT."""
        payload = credentials.get("payload", {})
        expires_in = credentials.get("expires_in", 3600)

        # Add standard claims
        payload.update({
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "jti": str(uuid.uuid4())
        })

        # Generate token
        token = jwt.encode(
            payload,
            self.jwt_secret,
            algorithm=self.jwt_algorithm
        )

        return {
            "token": token,
            "expires_in": expires_in
        }

    async def _authenticate_api_key(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using API key."""
        api_key = credentials.get("api_key")

        if not api_key:
            raise ValueError("Missing API key")

        # Validate API key
        if api_key not in self.api_keys:
            raise ValueError("Invalid API key")

        # Check permissions
        permissions = self.api_keys[api_key].get("permissions", [])

        return {
            "api_key": api_key,
            "permissions": permissions
        }

    async def _handle_refresh_token(self, message: Message) -> None:
        """Handle token refresh request."""
        try:
            session_id = message.data.get("session_id")

            if session_id not in self.sessions:
                raise ValueError("Invalid session ID")

            session = self.sessions[session_id]
            if session["auth_type"] != "oauth2":
                raise ValueError("Session is not OAuth2")

            result = await self._refresh_oauth2_token(session_id)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Token refresh error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="token_refresh_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _refresh_oauth2_token(self, session_id: str) -> Dict[str, Any]:
        """Refresh OAuth2 token."""
        session = self.sessions[session_id]
        refresh_token = session["credentials"].get("refresh_token")

        if not refresh_token:
            raise ValueError("No refresh token available")

        # Get client
        client_id = session["credentials"].get("client_id")
        client = self.oauth2_clients.get(client_id)

        if not client:
            raise ValueError("OAuth2 client not found")

        # Refresh token
        token = await client.refresh_token(refresh_token)

        # Update session
        session["credentials"].update({
            "access_token": token["access_token"],
            "refresh_token": token.get("refresh_token"),
            "expires_in": token.get("expires_in"),
            "token_type": token.get("token_type", "Bearer")
        })

        return session["credentials"]

    async def _handle_validate_session(self, message: Message) -> None:
        """Handle session validation request."""
        try:
            session_id = message.data.get("session_id")

            if session_id not in self.sessions:
                raise ValueError("Invalid session ID")

            session = self.sessions[session_id]
            if time.time() - session["created_at"] > self.session_ttl:
                raise ValueError("Session expired")

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"valid": True, "session": session}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Session validation error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="session_validation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_revoke_session(self, message: Message) -> None:
        """Handle session revocation request."""
        try:
            session_id = message.data.get("session_id")

            if session_id not in self.sessions:
                raise ValueError("Invalid session ID")

            await self._revoke_session(session_id)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "success"}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Session revocation error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="session_revocation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _revoke_session(self, session_id: str) -> None:
        """Revoke a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Revoke OAuth2 token if applicable
            if session["auth_type"] == "oauth2":
                try:
                    client_id = session["credentials"].get("client_id")
                    client = self.oauth2_clients.get(client_id)
                    if client:
                        await client.revoke_token(session["credentials"]["access_token"])
                except Exception as e:
                    self.logger.error(f"Error revoking OAuth2 token: {str(e)}")

            # Remove session
            del self.sessions[session_id]

    def _encrypt_session_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt session data."""
        try:
            # Convert data to JSON
            json_data = json.dumps(data)

            # Generate IV
            iv = os.urandom(16)

            # Create cipher
            cipher = AES.new(
                self.session_encryption_key.encode(),
                AES.MODE_CBC,
                iv
            )

            # Pad data
            padded_data = json_data.encode()
            padded_data += b'\0' * (16 - len(padded_data) % 16)

            # Encrypt
            encrypted_data = cipher.encrypt(padded_data)

            return {
                "data": base64.b64encode(encrypted_data).decode(),
                "iv": base64.b64encode(iv).decode()
            }
        except Exception as e:
            self.logger.error(f"Error encrypting session data: {str(e)}")
            raise

    def _decrypt_session_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt session data."""
        try:
            # Decode data
            data = base64.b64decode(encrypted_data["data"])
            iv = base64.b64decode(encrypted_data["iv"])

            # Create cipher
            cipher = AES.new(
                self.session_encryption_key.encode(),
                AES.MODE_CBC,
                iv
            )

            # Decrypt
            decrypted_data = cipher.decrypt(data)

            # Remove padding
            decrypted_data = decrypted_data.rstrip(b'\0')

            # Parse JSON
            return json.loads(decrypted_data.decode())
        except Exception as e:
            self.logger.error(f"Error decrypting session data: {str(e)}")
            raise

    async def _handle_get_session_info(self, message: Message) -> None:
        """Handle session info request."""
        try:
            session_id = message.data.get("session_id")

            if session_id not in self.sessions:
                raise ValueError("Invalid session ID")

            # Decrypt session data
            session_data = self._decrypt_session_data(self.sessions[session_id])

            # Update last activity
            session_data["last_activity"] = time.time()
            self.sessions[session_id] = self._encrypt_session_data(session_data)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={
                    "session_id": session_id,
                    "auth_type": session_data["auth_type"],
                    "created_at": session_data["created_at"],
                    "last_activity": session_data["last_activity"],
                    "ip_address": session_data.get("ip_address"),
                    "user_agent": session_data.get("user_agent")
                }
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error getting session info: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="session_info_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        # Add message to outbox
        self.outbox.put(message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.AUTHENTICATE:
            return await self._authenticate_basic(task.parameters)
        elif task.type == TaskType.REFRESH_TOKEN:
            session_id = task.parameters.get("session_id")
            return await self._refresh_oauth2_token(session_id)
        elif task.type == TaskType.VALIDATE_SESSION:
            session_id = task.parameters.get("session_id")
            if session_id not in self.sessions:
                raise ValueError("Invalid session ID")
            session = self.sessions[session_id]
            return {"valid": True, "session": session}
        elif task.type == TaskType.REVOKE_SESSION:
            session_id = task.parameters.get("session_id")
            await self._revoke_session(session_id)
            return {"status": "success"}
        else:
            raise ValueError(f"Unsupported task type for authentication agent: {task.type}")