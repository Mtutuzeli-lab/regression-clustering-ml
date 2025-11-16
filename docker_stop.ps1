# Docker Stop Script for Windows PowerShell
# This script stops and removes all Docker containers

Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host "STOPPING DOCKER CONTAINERS"
Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host ""

Write-Host "Stopping all containers..."
docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "All containers stopped successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start again, run: .\docker_run.ps1" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error stopping containers" -ForegroundColor Red
    exit 1
}
