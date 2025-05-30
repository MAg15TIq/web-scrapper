# 🚀 Deployment Guide - Multi-Agent Web Scraping System

This guide provides comprehensive instructions for deploying the Multi-Agent Web Scraping System to production environments.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 10GB+ available space
- **Network**: Stable internet connection

### Required Software

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git
sudo apt-get install -y tesseract-ocr redis-server postgresql

# macOS (using Homebrew)
brew install python git tesseract redis postgresql

# Windows (using Chocolatey)
choco install python git tesseract redis postgresql
```

## 🌍 Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/MAg15TIq/web-scrapper.git
cd web-scrapper
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional system dependencies
python -m spacy download en_core_web_sm
playwright install
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env  # or use your preferred editor
```

**Required Environment Variables:**

```bash
# Security
SECRET_KEY=your-super-secret-key-here
OPENAI_API_KEY=your-openai-api-key

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/webscraper
REDIS_URL=redis://localhost:6379/0

# Web Server
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_DEBUG=false
```

## 🏠 Local Development

### 1. Database Setup

```bash
# PostgreSQL setup
sudo -u postgres createuser webscraper
sudo -u postgres createdb webscraper
sudo -u postgres psql -c "ALTER USER webscraper PASSWORD 'your_password';"

# Run database migrations
alembic upgrade head
```

### 2. Start Services

```bash
# Start Redis (if not running as service)
redis-server

# Start the web API
python web/api/main.py

# In another terminal, start the CLI
python -m cli.enhanced_interface --interactive
```

### 3. Verify Installation

```bash
# Test API health
curl http://localhost:8000/health

# Test CLI
python -m cli.enhanced_interface agents

# Access API documentation
# Open browser: http://localhost:8000/api/docs
```

## 🏭 Production Deployment

### 1. System Configuration

```bash
# Create system user
sudo useradd -m -s /bin/bash webscraper
sudo usermod -aG sudo webscraper

# Create application directory
sudo mkdir -p /opt/webscraper
sudo chown webscraper:webscraper /opt/webscraper
```

### 2. Application Setup

```bash
# Switch to application user
sudo su - webscraper

# Clone and setup application
cd /opt/webscraper
git clone https://github.com/MAg15TIq/web-scrapper.git .
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Production Configuration

```bash
# Create production environment file
cat > .env << EOF
ENVIRONMENT=production
WEB_DEBUG=false
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgresql://webscraper:password@localhost:5432/webscraper_prod
REDIS_URL=redis://localhost:6379/0
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_WORKERS=4
EOF
```

### 4. Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/webscraper.service << EOF
[Unit]
Description=Multi-Agent Web Scraping System
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=webscraper
Group=webscraper
WorkingDirectory=/opt/webscraper
Environment=PATH=/opt/webscraper/venv/bin
ExecStart=/opt/webscraper/venv/bin/gunicorn web.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable webscraper
sudo systemctl start webscraper
```

### 5. Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt-get install nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/webscraper << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/webscraper /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🐳 Docker Deployment

### 1. Create Dockerfile

```dockerfile
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN python -m spacy download en_core_web_sm
RUN playwright install --with-deps

# Copy project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "web.api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

### 2. Docker Compose

```yaml
# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://webscraper:password@db:5432/webscraper
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=webscraper
      - POSTGRES_USER=webscraper
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web

volumes:
  postgres_data:
  redis_data:
EOF
```

### 3. Deploy with Docker

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale web service
docker-compose up -d --scale web=3
```

## ☁️ Cloud Deployment

### AWS Deployment

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Deploy using AWS ECS or EC2
# (Detailed AWS deployment guide would go here)
```

### Google Cloud Platform

```bash
# Install Google Cloud SDK
# Deploy using Google Cloud Run or Compute Engine
# (Detailed GCP deployment guide would go here)
```

### Azure Deployment

```bash
# Install Azure CLI
# Deploy using Azure Container Instances or App Service
# (Detailed Azure deployment guide would go here)
```

## 📊 Monitoring & Maintenance

### 1. Log Management

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/webscraper << EOF
/opt/webscraper/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 webscraper webscraper
}
EOF
```

### 2. Health Monitoring

```bash
# Create health check script
cat > /opt/webscraper/health_check.sh << EOF
#!/bin/bash
response=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ \$response -eq 200 ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP \$response)"
    exit 1
fi
EOF

chmod +x /opt/webscraper/health_check.sh

# Add to crontab for monitoring
echo "*/5 * * * * /opt/webscraper/health_check.sh" | crontab -
```

### 3. Backup Strategy

```bash
# Database backup script
cat > /opt/webscraper/backup.sh << EOF
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
pg_dump webscraper_prod > /opt/webscraper/backups/db_backup_\$DATE.sql
find /opt/webscraper/backups -name "*.sql" -mtime +7 -delete
EOF

chmod +x /opt/webscraper/backup.sh

# Schedule daily backups
echo "0 2 * * * /opt/webscraper/backup.sh" | crontab -
```

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Test connection
   psql -h localhost -U webscraper -d webscraper
   ```

3. **Redis Connection Issues**
   ```bash
   # Check Redis status
   sudo systemctl status redis
   
   # Test connection
   redis-cli ping
   ```

4. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R webscraper:webscraper /opt/webscraper
   sudo chmod +x /opt/webscraper/venv/bin/*
   ```

### Performance Optimization

1. **Database Optimization**
   ```sql
   -- Add indexes for better performance
   CREATE INDEX idx_jobs_status ON jobs(status);
   CREATE INDEX idx_jobs_created_at ON jobs(created_at);
   ```

2. **Redis Configuration**
   ```bash
   # Edit Redis config for production
   sudo nano /etc/redis/redis.conf
   # Set: maxmemory 2gb
   # Set: maxmemory-policy allkeys-lru
   ```

3. **Nginx Optimization**
   ```nginx
   # Add to nginx.conf
   gzip on;
   gzip_types text/plain application/json;
   client_max_body_size 100M;
   ```

## 📞 Support

For deployment issues:
- 📧 Email: support@webscraper.com
- 🐛 Issues: [GitHub Issues](https://github.com/MAg15TIq/web-scrapper/issues)
- 📖 Documentation: [Wiki](https://github.com/MAg15TIq/web-scrapper/wiki)

---

**Happy Deploying! 🚀**
