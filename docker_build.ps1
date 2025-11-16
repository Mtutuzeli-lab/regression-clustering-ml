# Docker Build Script for Windows PowerShell
# This script builds the Docker images for the ML project

Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host "BUILDING DOCKER IMAGES"
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

# Build the images
Write-Host "Building Docker images (this may take 5-10 minutes)..."
Write-Host ""

docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host "BUILD SUCCESSFUL" -ForegroundColor Green
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Run: .\docker_run.ps1" -ForegroundColor Cyan
    Write-Host "2. Access applications:" -ForegroundColor Cyan
    Write-Host "   - Streamlit: http://localhost:8501"
    Write-Host "   - MLflow: http://localhost:5000"
    Write-Host "   - TensorBoard: http://localhost:6006"
} else {
    Write-Host ""
    Write-Host "BUILD FAILED" -ForegroundColor Red
    Write-Host "Check the error messages above for details."
    exit 1
}
