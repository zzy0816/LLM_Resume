# -----------------------------
# Step 0: Define paths
# -----------------------------
$ProjectRoot = "D:\project\LLM_Resume"
$LogDir = Join-Path $ProjectRoot "logs"   # 指向实际日志目录

# -----------------------------
# Step 1: Ensure logs directory exists
# -----------------------------
Write-Output "[INIT] Ensure logs directory exists at $LogDir"
if (-Not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

# -----------------------------
# Step 2: Start ELK containers
# -----------------------------
Write-Output "[INIT] Starting ELK containers..."
Set-Location "$ProjectRoot\elk"
docker-compose up -d

# -----------------------------
# Step 3: Wait for services
# -----------------------------
Write-Output "[INIT] Waiting for services (30s)..."
Start-Sleep -Seconds 30

# -----------------------------
# Step 4: Check Elasticsearch status
# -----------------------------
Write-Output "[INIT] Checking Elasticsearch status..."
try {
    Invoke-WebRequest -Uri "http://localhost:9200" -UseBasicParsing | Out-Null
} catch {
    Write-Output "[ERROR] Elasticsearch not running. Please check docker logs elasticsearch"
    exit 1
}

# -----------------------------
# Step 5: Register Kibana index pattern
# -----------------------------
Write-Output "[INIT] Registering Kibana index pattern..."
try {
    Invoke-RestMethod -Uri "http://localhost:5601/api/saved_objects/index-pattern" `
        -Method Post `
        -Headers @{ "kbn-xsrf" = "true"; "Content-Type" = "application/json" } `
        -Body '{"attributes":{"title":"app-logs-*","timeFieldName":"timestamp"}}'
} catch {
    Write-Output "[WARN] Index pattern may already exist"
}

# -----------------------------
# Step 6: Run Python script to generate logs
# -----------------------------
Write-Output "[INIT] Running Python script..."
Set-Location $ProjectRoot
python ".\app\pipline\main.py"

Write-Output "[DONE] Initialization complete!"
Write-Output "Open Kibana: http://localhost:5601"
Write-Output "In Discover, select index pattern: app-logs-*"
