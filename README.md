# MemoryGraph

MemoryGraph is a persistent-memory, event-driven GraphRAG system designed to extract documents into knowledge graphs and vector stores, support grounded hybrid retrieval, and maintain long-term user memory asynchronously.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Streamlit Frontend (UI)   |
                      |         Port 8501           |
                      +--------------+--------------+
                                     |
                                     v HTTP
                      +-----------------------------+
                      |     FastAPI Backend API     |
                      |     Hybrid GraphRAG Engine  |
                      |          Port 8000          |
                      +--+-----------+-----------+--+
                         |           |           |
       +-----------------+           |           +-----------------+
       v                             v                             v
+-------------+               +-------------+               +-------------+
|    MySQL    |               |    Redis    |               | Apache Kafka|
| (Port 3306) |               | (Port 6379) |               | (Port 9092) |
| Documents,  |               |  Session    |               | Asynchronous|
| Graph Nodes |               |  Memory &   |               | Event Stream|
| & Edge Log  |               | Cache Store |               +------+------+
+------+------+               +-------------+                      |
       |                                                           v
       |                                                    +-------------+
       |                                                    | Doc Worker  |
       |                                                    | Extraction, |
       |                                                    | Embeddings, |
       |                                                    | Graph Index |
       |                                                    +------+------+
       |                                                           |
       +<------------------ Knowledge Graph -----------------------+
                           (NetworkX & MySQL)
```

---

## 🐳 Running with Docker (Local Deployment)

MemoryGraph includes a full `docker-compose.yml` that orchestrates the pre-downloaded images (`redis:latest`, `mysql:latest`, `apache/kafka:latest`) alongside the containerized FastAPI backend, Kafka document worker, and Streamlit frontend.

### 1. Launch All Containers
```bash
docker compose up --build -d
```

### 2. Verify Container Status
```bash
docker compose ps
```

Services exposed:
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Status**: [http://localhost:8000/health](http://localhost:8000/health)
- **MySQL Database**: `localhost:3306` (Database: `memorygraph`, User: `memorygraph`, Password: `memorygraph_secret`)
- **Redis Cache**: `localhost:6379`
- **Apache Kafka**: `localhost:9092` (KRaft single-node mode)

### 3. Stop All Containers
```bash
docker compose down
```

---

## ☁️ Deployment Strategy: Render vs. Vercel

### 🚀 Deploying to Render (Recommended for Containers)
Render natively supports stateful Docker containers, background workers, and persistent web services.

1. **Deploy with Blueprint (`render.yaml`)**:
   - Push your code to GitHub / GitLab.
   - Go to [dashboard.render.com](https://dashboard.render.com) -> **New** -> **Blueprint**.
   - Connect your repository. Render automatically reads `render.yaml` to spin up:
     - `memorygraph-api`: FastAPI Docker Web Service
     - `memorygraph-frontend`: Streamlit Docker Web Service
     - `memorygraph-worker`: Kafka Background Worker Docker container
     - `memorygraph-redis`: Managed Redis service on Render
2. **Cloud Database & Kafka Options for Production**:
   - **MySQL**: Connect to [Aiven MySQL](https://aiven.io), [PlanetScale](https://planetscale.com), [TiDB Cloud](https://tidbcloud.com), or [Railway MySQL].
   - **Kafka**: Connect to [Upstash Kafka](https://upstash.com/docs/kafka) (Serverless Kafka) or [Aiven Kafka] / [Confluent Cloud].

### ⚡ Deploying to Vercel (Frontend / Serverless)
**Important Note:** Vercel is a **serverless platform** designed for stateless web frontends (Next.js, React, Svelte) and short-lived edge functions.
- **What CANNOT run on Vercel**: Docker Compose, persistent background Kafka consumers, Streamlit (which requires a persistent WebSocket server), or MySQL/Redis container daemon instances.
- **How to use Vercel in this stack**:
  - Deploy the backend (`FastAPI`), `Worker`, `MySQL`, `Redis`, and `Kafka` to **Render** or managed cloud providers.
  - If you build a Next.js / React web interface, deploy that frontend to **Vercel** and set:
    ```env
    NEXT_PUBLIC_API_URL=https://memorygraph-api.onrender.com
    ```

---

## 💻 Local Development (Without Docker)

### 1. Requirements & Setup
Ensure Python 3.11+ is installed.
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```
