#!/bin/bash

# CodeWise Startup Script

echo -e "\033[32mStarting CodeWise AI Development Assistant...\033[0m"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "\033[33mCreating .env file from env.example...\033[0m"
    cp env.example .env
    echo -e "\033[31mPlease edit .env and add your OpenAI API key!\033[0m"
    echo -e "\033[31mThen run this script again.\033[0m"
    exit 1
fi

# Check if Docker is running
if ! docker version &> /dev/null; then
    echo -e "\033[31mDocker is not running. Please start Docker and try again.\033[0m"
    exit 1
fi

echo -e "\033[33mBuilding and starting containers...\033[0m"
docker-compose up --build

echo -e "\n\033[33mCodeWise has been stopped.\033[0m" 