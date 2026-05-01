#!/bin/bash
# ============================================================
#  deploy_aws.sh — Deploy Hospital System to AWS EC2
#  Run this ON your EC2 instance after SSH-ing in
#
#  STEP BY STEP:
#  1. Launch EC2 (instructions below)
#  2. SSH into instance
#  3. Run: bash deploy_aws.sh
# ============================================================

# ── STEP 1: Update system ─────────────────────────────────
sudo apt-get update -y
sudo apt-get upgrade -y

# ── STEP 2: Install Docker ────────────────────────────────
sudo apt-get install -y docker.io docker-compose-plugin curl git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu   # so you don't need sudo for docker

echo "✅ Docker installed"

# ── STEP 3: Clone your GitHub repo ───────────────────────
# Replace with your actual GitHub repo URL
git clone https://github.com/Ramit-Gupta23/hospital-los-system.git
cd hospital-los-system

echo "✅ Repo cloned"

# ── STEP 4: Add your dataset ─────────────────────────────
# Option A: upload from local machine (run this on YOUR machine, not EC2):
#   scp -i your-key.pem data/LengthOfStay.csv ubuntu@YOUR_EC2_IP:~/hospital-los-system/data/
#
# Option B: if dataset is public, wget it here
# wget -O data/LengthOfStay.csv "YOUR_DATASET_URL"

echo "⚠️  Make sure LengthOfStay.csv is in the data/ folder before continuing"
echo "⚠️  Also make sure your trained model is in models/ folder"

# ── STEP 5: Run Phase 1 to generate model (if not already done) ──
# python hospital_pipeline_phase1.py
# (only needed if models/ folder is empty)

# ── STEP 6: Start everything with Docker Compose ─────────
sudo docker compose up --build -d

echo ""
echo "=================================================="
echo "  DEPLOYMENT COMPLETE"
echo "=================================================="
echo "  FastAPI  → http://$(curl -s ifconfig.me):8000"
echo "  API Docs → http://$(curl -s ifconfig.me):8000/docs"
echo "  Dashboard→ http://$(curl -s ifconfig.me):8501"
echo "=================================================="
echo ""
echo "  Useful commands:"
echo "  View logs      : docker compose logs -f"
echo "  Stop services  : docker compose down"
echo "  Restart        : docker compose restart"
echo "  Check status   : docker compose ps"
