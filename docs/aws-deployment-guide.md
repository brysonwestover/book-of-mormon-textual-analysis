# AWS Deployment Guide for Robustness Analysis

## Instance Recommendation

| Instance | vCPUs | RAM | Price/hr | Est. Runtime | Est. Cost |
|----------|-------|-----|----------|--------------|-----------|
| c5.9xlarge | 36 | 72 GB | $1.53 | ~15 hours | ~$23 |
| **c5.18xlarge** | 72 | 144 GB | $3.06 | ~7-8 hours | ~$21-25 |
| c5.24xlarge | 96 | 192 GB | $4.08 | ~5-6 hours | ~$20-25 |

**Recommended: c5.18xlarge** - Best balance of speed and cost.

## Step-by-Step Setup

### 1. Launch EC2 Instance

1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Click **Launch Instance**
3. Configure:
   - **Name**: `bom-robustness-analysis`
   - **AMI**: Ubuntu Server 22.04 LTS (free tier eligible AMI is fine)
   - **Instance type**: `c5.18xlarge`
   - **Key pair**: Create new or select existing (you'll need this to SSH)
   - **Storage**: 20 GB gp3 (default is fine)
   - **Security group**: Allow SSH (port 22) from your IP

4. Click **Launch Instance**

### 2. Connect to Instance

```bash
# Replace with your key file and instance public IP
ssh -i "your-key.pem" ubuntu@<INSTANCE_PUBLIC_IP>
```

### 3. Upload Project Files

From your LOCAL machine (not the EC2 instance):

```bash
# Create a zip of the project (excluding venv and large files)
cd /home/bryson/book-of-mormon-textual-analysis
zip -r bom-analysis.zip . -x "venv/*" -x ".git/*" -x "*.pyc" -x "__pycache__/*"

# Upload to EC2
scp -i "your-key.pem" bom-analysis.zip ubuntu@<INSTANCE_PUBLIC_IP>:~
```

### 4. Setup on EC2

SSH into the instance and run:

```bash
# Unzip project
unzip bom-analysis.zip -d bom-analysis
cd bom-analysis

# Install dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy scipy scikit-learn joblib
```

### 5. Run the Analysis

```bash
# Start a screen session (persists if SSH disconnects)
screen -S robustness

# Activate venv and run
source venv/bin/activate
python scripts/run_robustness_optimized.py --permutations 10000

# To detach from screen: Press Ctrl+A, then D
# To reattach later: screen -r robustness
```

### 6. Monitor Progress

```bash
# In another terminal or after reattaching
tail -f results/robustness-checkpoint-v2.json

# Or check the output directly
# Progress shows every 500 permutations
```

### 7. Download Results

After completion, from your LOCAL machine:

```bash
scp -i "your-key.pem" ubuntu@<INSTANCE_PUBLIC_IP>:~/bom-analysis/results/robustness-results.json ./results/
```

### 8. IMPORTANT: Terminate Instance

**Don't forget to terminate the instance when done to avoid charges!**

1. Go to EC2 Console
2. Select your instance
3. Actions → Instance State → Terminate

## Cost Control Tips

- Use **Spot Instances** for ~70% savings ($0.92/hr instead of $3.06/hr for c5.18xlarge)
  - Risk: Instance can be interrupted with 2-min warning
  - Mitigation: Script has checkpointing, can resume if interrupted

- Set a **billing alarm** in AWS to alert you at a threshold (e.g., $30)

## Spot Instance Launch (Optional - Cheaper)

When launching, instead of on-demand:
1. Check "Request Spot Instances" under Advanced details
2. Set maximum price (e.g., $1.50/hr for c5.18xlarge)
3. The script's checkpointing will save progress if interrupted

## Troubleshooting

**SSH connection refused:**
- Check security group allows SSH from your IP
- Verify instance is running

**Out of memory:**
- Unlikely with c5.18xlarge (144 GB RAM)
- If using smaller instance, reduce `--jobs` parameter

**Script crashes:**
- Check `results/robustness-checkpoint-v2.json` for progress
- Restart without `--no-checkpoint` to resume
