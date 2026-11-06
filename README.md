# Sumati IAS RAG Python Server

Production setup for the Flask-based RAG API, served with Waitress and managed by `systemd` on Ubuntu.

## Configuration

| Setting | Value |
|---|---|
| Project directory | `/var/www/sumati-ias-rag-chain` |
| Application entry point | `server.py` |
| Python environment | `.venv` |
| systemd service | `sumati-ias-rag-chain` |
| Bind address | `127.0.0.1` |
| Port | `8000` |
| WSGI server | Waitress |

The API listens only on localhost. Your Node/Express backend should call `http://127.0.0.1:8000`.

## 1. Create and install the virtual environment

Choose the Python version supported by your dependencies. The following example uses Python 3.12:

```bash
cd /var/www/sumati-ias-rag-chain
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python --version
which python
```

Install project dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install waitress
```

If you do not have a `requirements.txt` yet, install the packages your application needs and generate one:

```bash
pip freeze > requirements.txt
```

Always verify that `python` and `pip` point inside this project's `.venv`.

## 2. Serve Flask with Waitress

Do not use Flask's development server for production. At the bottom of `server.py`, start Waitress:

```python
from waitress import serve

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=8000, threads=8)
```

This assumes your Flask application instance is named `app`.

## 3. Configure environment variables

Keep secrets in an environment file or another secure secrets manager. Do not commit `.env` to Git.

Example `.env` entries (replace the placeholders):

```dotenv
MONGODB_URI=your_mongodb_connection_string
# Add the other environment variables required by server.py
```

If the application loads `.env` itself, ensure `python-dotenv` is installed and loaded by the application. Restrict file permissions:

```bash
chmod 600 .env
```

Add these entries to `.gitignore`:

```gitignore
.venv/
.env
__pycache__/
*.py[cod]
```

## 4. Create the systemd service

Create the service file:

```bash
sudo nano /etc/systemd/system/sumati-ias-rag-chain.service
```

Use this configuration:

```ini
[Unit]
Description=Sumati IAS RAG Python Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/sumati-ias-rag-chain
ExecStart=/var/www/sumati-ias-rag-chain/.venv/bin/python /var/www/sumati-ias-rag-chain/server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

The example runs as `root` to match the current deployment setup. For a hardened production deployment, use a dedicated service account and grant it ownership/access only to the application files and required resources.

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sumati-ias-rag-chain
sudo systemctl start sumati-ias-rag-chain
sudo systemctl status sumati-ias-rag-chain
```

## 5. Service management

```bash
# Start
sudo systemctl start sumati-ias-rag-chain

# Stop
sudo systemctl stop sumati-ias-rag-chain

# Restart
sudo systemctl restart sumati-ias-rag-chain

# Check status
sudo systemctl status sumati-ias-rag-chain

# View the latest logs
sudo journalctl -u sumati-ias-rag-chain -n 100 --no-pager

# Follow logs live
sudo journalctl -u sumati-ias-rag-chain -f
```

## 6. Deploy application changes

```bash
cd /var/www/sumati-ias-rag-chain
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart sumati-ias-rag-chain
sudo systemctl status sumati-ias-rag-chain
```

If dependencies changed, make sure `requirements.txt` is updated and committed with the application.

## 7. Troubleshooting: port 8000 is already in use

Check which process owns the port:

```bash
sudo ss -ltnp | grep ':8000'
sudo lsof -iTCP:8000 -sTCP:LISTEN -n -P
```

Inspect the reported PID before stopping anything:

```bash
ps -fp <PID>
sudo cat /proc/<PID>/cgroup
```

Only stop the process after confirming it is an old copy of this RAG server. Do not kill all Python processes; the operating system may be running unrelated Python services.

### If PM2 is also managing this Python server

A PM2-managed copy can restart automatically after you kill its PID, causing a port conflict with systemd. List PM2 processes:

```bash
pm2 list
pm2 describe <id>
```

Confirm the entry points to this project's `server.py`, then remove only that Python entry:

```bash
pm2 delete <id>
pm2 save
```

Do **not** run `pm2 kill` if PM2 is managing other Node.js applications. Once the duplicate is removed, start the systemd service:

```bash
sudo systemctl reset-failed sumati-ias-rag-chain
sudo systemctl start sumati-ias-rag-chain
sudo systemctl status sumati-ias-rag-chain
```

## 8. Node.js backend integration

Keep the Python service bound to `127.0.0.1:8000`. From the Node.js backend on the same VPS, send requests to:

```text
http://127.0.0.1:8000
```

Do not expose port `8000` publicly through UFW or Nginx unless you have a specific, secured reason to do so. Prefer routing authenticated requests through the Node.js backend.

## 9. Production checklist

- [ ] `server.py` uses Waitress rather than Flask's development server.
- [ ] systemd uses the project's `.venv/bin/python` interpreter.
- [ ] The service is enabled and starts successfully after reboot.
- [ ] Only one process listens on `127.0.0.1:8000`.
- [ ] PM2 does not also supervise this Python server.
- [ ] Secrets are not committed to Git and `.env` permissions are restricted.
- [ ] Node.js calls the internal localhost API.
- [ ] Logs are checked after deployment and after dependency changes.
