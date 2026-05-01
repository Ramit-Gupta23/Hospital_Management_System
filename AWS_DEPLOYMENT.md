# AWS EC2 Deployment Guide
## Hospital Management System — Phase 4

---

## Step 1: Launch EC2 Instance (Free Tier)

1. Go to **https://aws.amazon.com** → sign in → EC2 → Launch Instance
2. Settings:
   - **Name**: hospital-los-system
   - **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance type**: t2.micro (Free tier — 1 vCPU, 1GB RAM)
   - **Key pair**: Create new → download the `.pem` file → save it safely
   - **Security group** — Add these inbound rules:
     | Type       | Port | Source    |
     |------------|------|-----------|
     | SSH        | 22   | My IP     |
     | Custom TCP | 8000 | Anywhere  |
     | Custom TCP | 8501 | Anywhere  |
   - **Storage**: 20GB (default is fine)
3. Click **Launch Instance**
4. Wait ~2 minutes for instance to start
5. Copy the **Public IPv4 address** from the EC2 dashboard

---

## Step 2: SSH Into Your Instance

On your local machine (Windows — use PowerShell or Git Bash):

```bash
# Give key file correct permissions (Mac/Linux)
chmod 400 your-key.pem

# SSH in
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

**Windows users**: Use Git Bash or WSL, or use PuTTY with the .pem converted to .ppk

---

## Step 3: Upload Your Files

In a NEW terminal on your local machine (not SSH):

```bash
# Upload your entire project folder to EC2
scp -i your-key.pem -r "C:\Users\ramit\Desktop\Hospital Project" ubuntu@YOUR_EC2_IP:~/hospital-los-system

# Or upload individual files:
scp -i your-key.pem data/LengthOfStay.csv ubuntu@YOUR_EC2_IP:~/hospital-los-system/data/
scp -i your-key.pem models/best_model_XGBoost.joblib ubuntu@YOUR_EC2_IP:~/hospital-los-system/models/
```

---

## Step 4: Deploy on EC2

Back in your SSH terminal:

```bash
# Go to project folder
cd hospital-los-system

# Run the deploy script
bash deploy_aws.sh
```

This installs Docker, builds both containers, and starts everything.

---

## Step 5: Access Your Live System

After deploy script finishes:

| Service    | URL                                    |
|------------|----------------------------------------|
| FastAPI    | `http://YOUR_EC2_IP:8000`             |
| API Docs   | `http://YOUR_EC2_IP:8000/docs`        |
| Dashboard  | `http://YOUR_EC2_IP:8501`             |

**These are real public URLs** — anyone can access them.
Put `http://YOUR_EC2_IP:8000/docs` in your resume and GitHub README.

---

## Useful Commands After Deployment

```bash
# Check if containers are running
docker compose ps

# View live logs
docker compose logs -f

# View only API logs
docker compose logs api -f

# Restart everything
docker compose restart

# Stop everything
docker compose down

# Rebuild after code changes
docker compose up --build -d
```

---

## Free Tier Limits

- t2.micro gives you **750 hours/month free** for 12 months
- That's enough to run 24/7 for a full month
- **Stop the instance** when not using it to save hours
- Stop from EC2 dashboard → select instance → Instance State → Stop

---

## Troubleshooting

**Port not accessible**: Check Security Group inbound rules — ports 8000 and 8501 must allow 0.0.0.0/0

**Out of memory**: t2.micro has 1GB RAM. If it crashes, add swap:
```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Model not found**: Make sure `models/` folder has the `.joblib` file before running docker compose

---

## What to Put in Your README and Resume

```
Live Demo:
  API    : http://YOUR_EC2_IP:8000/docs
  Dashboard: http://YOUR_EC2_IP:8501
```

```
Deployed on AWS EC2 (Ubuntu 22.04, t2.micro)
Containerized with Docker Compose
Services: FastAPI inference API + Streamlit analytics dashboard
```
