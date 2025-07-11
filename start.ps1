# CodeWise Startup Script

Write-Host "Starting CodeWise AI Development Assistant..." -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from env.example..." -ForegroundColor Yellow
    Copy-Item "env.example" ".env"
    Write-Host "Please edit .env and add your OpenAI API key!" -ForegroundColor Red
    Write-Host "Then run this script again." -ForegroundColor Red
    exit 1
}

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

Write-Host "Building and starting containers..." -ForegroundColor Yellow
docker-compose up --build

Write-Host "`nCodeWise has been stopped." -ForegroundColor Yellow 