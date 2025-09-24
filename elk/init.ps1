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
# Step 3: Wait for Elasticsearch to be ready
# -----------------------------
Write-Output "[INIT] Waiting for Elasticsearch to start..."
$maxRetries = 12
$retryCount = 0
$esRunning = $false

do {
    Start-Sleep -Seconds 5
    try {
        Invoke-WebRequest -Uri "http://localhost:9200" -UseBasicParsing | Out-Null
        $esRunning = $true
    } catch {
        $retryCount++
        Write-Output "[INIT] Elasticsearch not ready yet, retry $retryCount/$maxRetries..."
    }
} while (-Not $esRunning -and $retryCount -lt $maxRetries)

if (-Not $esRunning) {
    Write-Output "[ERROR] Elasticsearch did not start. Check docker logs for details."
    exit 1
} else {
    Write-Output "[INIT] Elasticsearch is running."
}

# -----------------------------
# Step 4: Register Kibana index pattern
# -----------------------------
Write-Output "[INIT] Registering Kibana index pattern..."
try {
    $existing = Invoke-RestMethod -Uri "http://localhost:5601/api/saved_objects/index-pattern/app-logs-*" -Method Get -UseBasicParsing -ErrorAction SilentlyContinue
} catch {
    $existing = $null
}

if (-Not $existing) {
    try {
        Invoke-RestMethod -Uri "http://localhost:5601/api/saved_objects/index-pattern" `
            -Method Post `
            -Headers @{ "kbn-xsrf" = "true"; "Content-Type" = "application/json" } `
            -Body '{"attributes":{"title":"app-logs-*","timeFieldName":"timestamp"}}'
        Write-Output "[INIT] Index pattern 'app-logs-*' created."
    } catch {
        Write-Output "[WARN] Failed to create index pattern, it may already exist."
    }
} else {
    Write-Output "[INIT] Index pattern 'app-logs-*' already exists."
}

# -----------------------------
# Step 5: Run Python script to generate logs
# -----------------------------
Write-Output "[INIT] Running Python script to generate logs..."
Set-Location $ProjectRoot
try {
    python ".\app\test_tool\parser_test.py"
    Write-Output "[INIT] Python script executed successfully."
} catch {
    Write-Output "[WARN] Python script failed: $_"
}

# -----------------------------
# Step 6: Done
# -----------------------------
Write-Output "[DONE] Initialization complete!"
Write-Output "Open Kibana: http://localhost:5601"
Write-Output "In Discover, select index pattern: app-logs-*"
