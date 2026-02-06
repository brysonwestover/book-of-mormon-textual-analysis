# Session Progress: February 5, 2026

## Summary

Deployed Phase 2.A robustness analysis to AWS after discovering local execution would take ~40 days. Created supplementary TOST equivalence testing and documented all pre-registration deviations.

---

## AWS Instance (IMPORTANT - TERMINATE WHEN DONE)

| Parameter | Value |
|-----------|-------|
| Instance ID | `i-02fa719481b2bb537` |
| Instance Type | c7i.8xlarge (32 vCPUs) |
| Public IP | `54.84.56.89` |
| Region | us-east-1 |
| Cost | ~$1.43/hour |
| Started | ~3:05 PM MST |
| Expected completion | ~6-7 AM MST (Feb 6) |
| SSH Key | `~/.ssh/bom-analysis-key.pem` |
| AWS Profile | `brysonwestover` |

### Commands

**Check progress:**
```bash
ssh -i ~/.ssh/bom-analysis-key.pem ubuntu@54.84.56.89 "tail -20 ~/bom-analysis/robustness.log"
```

**Check if still running:**
```bash
ssh -i ~/.ssh/bom-analysis-key.pem ubuntu@54.84.56.89 "ps aux | grep python | grep -v grep | wc -l"
```

**Download results when complete:**
```bash
scp -i ~/.ssh/bom-analysis-key.pem ubuntu@54.84.56.89:~/bom-analysis/results/robustness-results.json ./results/
scp -i ~/.ssh/bom-analysis-key.pem ubuntu@54.84.56.89:~/bom-analysis/results/robustness-checkpoint-v2.json ./results/
```

**TERMINATE INSTANCE (after downloading results):**
```bash
~/.local/bin/aws ec2 terminate-instances --instance-ids i-02fa719481b2bb537 --profile brysonwestover --region us-east-1
```

**Hourly monitoring log:**
```bash
cat /tmp/aws_monitor.log
```

---

## Completed Today

### 1. Diagnosed Original Robustness Script Issue
- Original script would take ~40 days for 10,000 permutations
- Bottleneck: Re-computing features for every permutation + dense matrix conversion

### 2. Created Optimized Robustness Script
- `scripts/run_robustness_optimized.py` (v2.0.0)
- Key optimizations:
  - Pre-compute feature matrices once
  - Keep sparse matrices for n-grams
  - Parallelize with joblib
- Local test: 50 perms in 45 min (vs estimated ~4 hours with original)

### 3. Deployed to AWS
- Set up AWS CLI with SSO (`~/.local/bin/aws configure sso`)
- Created key pair, security group, launched c7i.8xlarge
- Uploaded code and started 10,000 permutation run

### 4. Created TOST Equivalence Testing
- `scripts/run_tost_equivalence.py`
- Tests whether classifier accuracy is statistically equivalent to chance
- Results in `results/tost-equivalence-report.md`
- Key finding: Near-equivalence (p=0.06), Bayes factors favor null (BF01≈2.5)

### 5. Documentation
- `docs/phase2-primary-study-summary.md` - Summary of Phase 2.0 findings
- `docs/tost-writeup-suggestions.md` - GPT-generated write-up language
- `docs/osf-amendment-2026-02-05.md` - Pre-registration deviations

---

## Key Findings So Far

### Phase 2.0 Primary Analysis (Complete)
- **Accuracy**: 24.2% (chance = 25%)
- **Permutation p-value**: 0.177
- **Interpretation**: No detectable stylometric signal

### TOST Supplementary Analysis (Complete)
- **TOST p-value**: 0.06 (near-equivalence)
- **Bayes Factor**: BF01 = 2.85 (weak evidence for null)
- **Interpretation**: Consistent with chance-level performance

### Phase 2.A Robustness (Running on AWS)
- 50-perm test showed: maxT corrected p = 0.31 (robust null)
- Full 10,000-perm test running now
- Expected result: Similar to 50-perm test

---

## Pending Tasks

1. **Download AWS results** when complete (~6-7 AM)
2. **Terminate AWS instance** to stop charges
3. **Write Phase 2.A Robustness Report** integrating maxT results
4. **Final write-up** combining all findings

---

## Files Created/Modified Today

### New Files
- `scripts/run_robustness_optimized.py`
- `scripts/run_tost_equivalence.py`
- `scripts/aws-deploy.sh`
- `docs/aws-deployment-guide.md`
- `docs/osf-amendment-2026-02-05.md`
- `docs/tost-writeup-suggestions.md`
- `docs/phase2-primary-study-summary.md`
- `results/tost-equivalence-results.json`
- `results/tost-equivalence-report.md`
- `~/.ssh/bom-analysis-key.pem` (AWS SSH key)

### AWS Resources Created
- Key pair: `bom-analysis-key`
- Security group: `sg-0eb10794f5c7acfc5` (bom-analysis-sg)
- EC2 instance: `i-02fa719481b2bb537`

---

## Cost Estimate

- Instance runtime: ~10 hours × $1.43/hr = **~$14-15**
- Storage: negligible
- **Total estimated**: ~$15

Remember to terminate the instance after downloading results!
