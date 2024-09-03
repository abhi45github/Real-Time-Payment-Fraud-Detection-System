# 🚀 Deployment Guide - Real-Time Fraud Detection System

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Hardware
- **CPU**: 8 cores (16 recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **Network**: 1Gbps

### Software Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.25+ (for K8s deployment)
- Python 3.10+
- PostgreSQL 15+
- Redis 7.0+
- Kafka 3.0+

---

## Local Development

### 1. Clone Repository
```bash
git clone https://github.com/machapraveen/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Local Services
```bash
# Start Redis
redis-server --port 6379

# Start PostgreSQL
pg_ctl start -D /usr/local/var/postgres

# Start Kafka
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh config/server.properties
```

### 4. Initialize Database
```bash
# Create database
createdb fraud_detection

# Run migrations
python src/database/migrations.py

# Generate test data
python scripts/generate_test_data.py
```

### 5. Train Model
```bash
python scripts/train_model.py
```

### 6. Start API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Docker Deployment

### 1. Build Images
```bash
# Build API image
docker build -f docker/Dockerfile -t fraud-detection:latest .

# Or use docker-compose
docker-compose -f docker/docker-compose.yml build
```

### 2. Start All Services
```bash
cd docker
docker-compose up -d
```

### 3. Verify Services
```bash
# Check service status
docker-compose ps

# Check logs
docker-compose logs -f fraud-api

# Test API
curl http://localhost:8000/api/v1/health
```

### 4. Initialize Data
```bash
# Run database migrations
docker exec fraud-api python src/database/migrations.py

# Train initial model
docker exec fraud-api python scripts/train_model.py
```

---

## Kubernetes Deployment

### 1. Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
```

### 2. Create Namespace
```bash
kubectl create namespace fraud-detection
```

### 3. Deploy Redis
```bash
kubectl apply -f kubernetes/redis-cluster.yaml
```

### 4. Deploy Kafka
```bash
# Using Strimzi operator
kubectl create -f 'https://strimzi.io/install/latest?namespace=fraud-detection'
kubectl apply -f kubernetes/kafka-cluster.yaml
```

### 5. Deploy PostgreSQL
```bash
# Using Helm
helm install postgres bitnami/postgresql \
  --namespace fraud-detection \
  --set auth.postgresPassword=fraud_pass \
  --set auth.database=fraud_detection
```

### 6. Deploy Application
```bash
# Create ConfigMap
kubectl create configmap fraud-config \
  --from-env-file=.env \
  -n fraud-detection

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
```

### 7. Setup Ingress
```bash
kubectl apply -f kubernetes/ingress.yaml
```

### 8. Verify Deployment
```bash
# Check pods
kubectl get pods -n fraud-detection

# Check services
kubectl get svc -n fraud-detection

# Port forward for testing
kubectl port-forward -n fraud-detection svc/fraud-api 8000:8000
```

---

## Cloud Deployment

### AWS Deployment

#### 1. Setup Infrastructure
```bash
# Install AWS CLI
aws configure

# Create EKS cluster
eksctl create cluster \
  --name fraud-detection \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3
```

#### 2. Deploy with Terraform
```hcl
# terraform/main.tf
module "fraud_detection" {
  source = "./modules/fraud-detection"

  cluster_name = "fraud-detection"
  region       = "us-west-2"

  # RDS for PostgreSQL
  db_instance_class = "db.t3.large"
  db_storage        = 100

  # ElastiCache for Redis
  cache_node_type = "cache.t3.medium"
  cache_nodes     = 3

  # MSK for Kafka
  kafka_version    = "3.0.0"
  kafka_instances  = 3
}
```

#### 3. Deploy Application
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_URI
docker build -t fraud-detection .
docker tag fraud-detection:latest $ECR_URI/fraud-detection:latest
docker push $ECR_URI/fraud-detection:latest

# Deploy to EKS
kubectl apply -f kubernetes/
```

### Google Cloud Deployment

#### 1. Setup GKE
```bash
# Create cluster
gcloud container clusters create fraud-detection \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n2-standard-4
```

#### 2. Deploy Services
```bash
# Cloud SQL for PostgreSQL
gcloud sql instances create fraud-db \
  --database-version=POSTGRES_15 \
  --tier=db-n1-standard-2

# Memorystore for Redis
gcloud redis instances create fraud-cache \
  --size=5 \
  --region=us-central1

# Deploy application
kubectl apply -f kubernetes/
```

### Azure Deployment

#### 1. Setup AKS
```bash
# Create resource group
az group create --name fraud-detection-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group fraud-detection-rg \
  --name fraud-detection-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3
```

#### 2. Deploy Services
```bash
# Azure Database for PostgreSQL
az postgres server create \
  --resource-group fraud-detection-rg \
  --name fraud-db \
  --sku-name GP_Gen5_2

# Azure Cache for Redis
az redis create \
  --resource-group fraud-detection-rg \
  --name fraud-cache \
  --sku Standard \
  --vm-size c3
```

---

## Production Checklist

### Security
- [ ] Enable TLS/SSL for all endpoints
- [ ] Rotate API keys and secrets
- [ ] Enable WAF (Web Application Firewall)
- [ ] Setup VPN/Private endpoints
- [ ] Enable audit logging
- [ ] Implement data encryption at rest
- [ ] Setup RBAC permissions

### Performance
- [ ] Configure auto-scaling (HPA/VPA)
- [ ] Setup CDN for static assets
- [ ] Enable database connection pooling
- [ ] Configure Redis cluster mode
- [ ] Optimize Kafka partitions
- [ ] Enable query caching

### Monitoring
- [ ] Deploy Prometheus/Grafana
- [ ] Configure alerting rules
- [ ] Setup log aggregation (ELK/CloudWatch)
- [ ] Enable distributed tracing
- [ ] Configure uptime monitoring
- [ ] Setup PagerDuty integration

### Backup & DR
- [ ] Configure database backups
- [ ] Setup cross-region replication
- [ ] Document recovery procedures
- [ ] Test failover scenarios
- [ ] Setup backup retention policy
- [ ] Create runbooks

### Compliance
- [ ] Enable PCI DSS compliance
- [ ] Configure GDPR data handling
- [ ] Setup audit trails
- [ ] Document data retention policies
- [ ] Enable encryption in transit
- [ ] Regular security scanning

---

## Monitoring & Maintenance

### Key Metrics to Monitor

#### Application Metrics
- Transaction throughput (target: 5000 TPS)
- Response time P99 (target: <100ms)
- Fraud detection accuracy (target: >95%)
- False positive rate (target: <5%)
- Cache hit rate (target: >70%)

#### System Metrics
- CPU utilization (<80%)
- Memory usage (<80%)
- Disk I/O
- Network throughput
- Database connections

### Maintenance Tasks

#### Daily
- Review error logs
- Check fraud detection metrics
- Monitor queue sizes
- Verify backup completion

#### Weekly
- Review performance trends
- Update fraud patterns
- Clean up old logs
- Review security alerts

#### Monthly
- Retrain ML models
- Update dependencies
- Performance optimization
- Security patches
- Capacity planning

---

## Troubleshooting

### Common Issues

#### High Latency
```bash
# Check Redis connection
redis-cli ping

# Monitor slow queries
redis-cli SLOWLOG GET 10

# Check database performance
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;

# Scale API pods
kubectl scale deployment fraud-api --replicas=5
```

#### Low Accuracy
```bash
# Retrain model with recent data
python scripts/train_model.py --use-recent-data

# Check feature drift
python scripts/check_feature_drift.py

# Update fraud patterns
python scripts/update_patterns.py
```

#### Service Unavailable
```bash
# Check pod status
kubectl get pods -n fraud-detection

# Check service endpoints
kubectl get endpoints -n fraud-detection

# Check logs
kubectl logs -f deployment/fraud-api

# Restart services
kubectl rollout restart deployment/fraud-api
```

### Performance Tuning

#### Database Optimization
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_transactions_user_time
ON transactions(user_id, transaction_time DESC);

-- Update statistics
ANALYZE transactions;

-- Check query performance
EXPLAIN ANALYZE SELECT ...;
```

#### Redis Optimization
```bash
# Increase max memory
CONFIG SET maxmemory 4gb

# Enable persistence
CONFIG SET appendonly yes

# Optimize eviction policy
CONFIG SET maxmemory-policy allkeys-lru
```

#### Kafka Optimization
```bash
# Increase partitions
kafka-topics.sh --alter --topic transactions --partitions 50

# Adjust retention
kafka-configs.sh --alter --entity-type topics --entity-name transactions \
  --add-config retention.ms=86400000
```

---

## Support

### Documentation
- API Docs: http://localhost:8000/docs
- Architecture: [docs/architecture.md](docs/architecture.md)
- Model Guide: [docs/model.md](docs/model.md)

### Contact
- Team: Fraud Detection ML Team
- Email: fraud-ml@example.com
- Slack: #fraud-detection

### Resources
- GitHub: https://github.com/machapraveen/fraud-detection-system
- Wiki: https://wiki.example.com/fraud-detection
- Runbooks: https://runbooks.example.com/fraud-detection

---

## License

Copyright (c) 2025 - All rights reserved.
# Update: 2025-10-01T15:32:00.121659 - 5659

# Update: 2025-10-01T15:32:00.258345 - 5673

# Update: 2025-10-01T15:32:00.681134 - 5779

# Update: 2025-10-01T15:32:00.775001 - 3910

# Update: 2025-10-01T15:32:02.299356 - 9359

# Update: 2025-10-01T15:32:02.361790 - 3766

# Update: 2025-10-01T15:32:04.474147 - 5657

# Update: 2025-10-01T15:32:04.665313 - 1487

# Update: 2025-10-01T15:32:04.887932 - 7535

# Update: 2025-10-01T15:32:05.585018 - 6648

# Update: 2025-10-01T15:32:05.914265 - 2465

# Update: 2025-10-01T15:32:05.975003 - 9566

# Update: 2025-10-01T15:32:06.319764 - 3128

# Update: 2025-10-01T15:32:06.381444 - 7252

# Update: 2025-10-01T15:32:06.586420 - 4014

# Update: 2025-10-01T15:32:06.839577 - 9006

# Update: 2025-10-01T15:32:08.218503 - 4927

# Update: 2025-10-01T15:32:08.282065 - 2122

# Update: 2025-10-01T15:32:08.471384 - 6491

# Update: 2025-10-01T15:32:08.679713 - 1263

# Update: 2025-10-01T15:32:10.238270 - 2850

# Update: 2025-10-01T15:32:10.445061 - 2608

# Update: 2025-10-01T15:32:11.959468 - 9657

# Update: 2025-10-01T15:32:12.211242 - 5485

# Update: 2025-10-01T15:32:12.338775 - 6425

# Update: 2025-10-01T15:32:13.133453 - 1443

# Update: 2025-10-01T15:32:14.157802 - 9070

# Update: 2025-10-01T15:32:16.416689 - 1167

# Update: 2025-10-01T15:32:16.543596 - 8876

# Update: 2025-10-01T15:32:18.124933 - 8635

# Update: 2025-10-01T15:32:18.521170 - 4446

# Update: 2025-10-01T15:32:19.155108 - 1702

# Update: 2025-10-01T15:32:20.198900 - 6795

# Update: 2025-10-01T15:32:20.263686 - 2028

# Update: 2025-10-01T15:32:21.042111 - 2165

# Update: 2025-10-01T15:32:22.212072 - 9224

# Update: 2025-10-01T15:32:22.974532 - 5720

# Update: 2025-10-01T15:32:23.980799 - 5471

# Update: 2025-10-01T15:32:24.550986 - 8881

# Update: 2025-10-01T15:32:26.011490 - 1313

# Update: 2025-10-01T15:32:26.265479 - 6826

# Update: 2025-10-01T15:32:26.611362 - 5819

# Update: 2025-10-01T15:32:27.799484 - 4555

# Update: 2025-10-01T15:32:28.134210 - 2185

# Update: 2025-10-01T15:32:28.198050 - 3439

# Update: 2025-10-01T15:32:28.589218 - 4993

# Update: 2025-10-01T15:32:28.651636 - 7251

# Update: 2025-10-01T15:32:28.843574 - 8090
