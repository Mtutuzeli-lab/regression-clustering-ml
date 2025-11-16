# Docker Run Script for Windows PowerShell
# This script starts all Docker containers

Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host "STARTING DOCKER CONTAINERS"
Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..."
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}
Write-Host "Docker is running" -ForegroundColor Green
Write-Host ""

# Start containers
Write-Host "Starting all containers..."
Write-Host ""

docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host "CONTAINERS STARTED SUCCESSFULLY" -ForegroundColor Green
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host ""
    Write-Host "Applications are now running:"
    Write-Host ""
    Write-Host "  Streamlit App:  http://localhost:8501" -ForegroundColor Cyan
    Write-Host "  MLflow UI:      http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  TensorBoard:    http://localhost:6006" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Useful commands:"
    Write-Host "  View logs:      docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "  Stop all:       docker-compose down" -ForegroundColor Yellow
    Write-Host "  Restart all:    docker-compose restart" -ForegroundColor Yellow
    Write-Host ""
    
    # Wait a moment and open browsers
    Write-Host "Opening applications in browser..."
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8501"
    Start-Process "http://localhost:5000"
    Start-Process "http://localhost:6006"
} else {
    Write-Host ""
    Write-Host "FAILED TO START CONTAINERS" -ForegroundColor Red
    Write-Host "Check the error messages above for details."
    exit 1
}
