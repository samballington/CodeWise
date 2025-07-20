#!/usr/bin/env python3
"""
Test script specifically for JavaScript AST chunker
"""

from ast_chunker import ASTChunker
from pathlib import Path

def test_javascript_ast():
    """Test JavaScript AST chunker with larger content"""
    chunker = ASTChunker()
    
    # Large JavaScript content to avoid small file chunker
    content = """import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { debounce } from 'lodash';

// Regular function declaration
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += item.price * item.quantity;
    }
    return total;
}

// Export function declaration
export function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Async function declaration
async function fetchUserData(userId) {
    try {
        const response = await axios.get(`/api/users/${userId}`);
        return response.data;
    } catch (error) {
        console.error('Error fetching user data:', error);
        throw error;
    }
}

// Arrow function assigned to const
const processData = (data) => {
    return data.map(item => ({
        ...item,
        processed: true,
        timestamp: Date.now()
    }));
};

// Another arrow function
const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

// Regular class declaration
class DataProcessor {
    constructor(options = {}) {
        this.options = options;
        this.cache = new Map();
    }
    
    process(data) {
        const key = this.generateKey(data);
        if (this.cache.has(key)) {
            return this.cache.get(key);
        }
        
        const result = this.performProcessing(data);
        this.cache.set(key, result);
        return result;
    }
    
    generateKey(data) {
        return JSON.stringify(data);
    }
    
    performProcessing(data) {
        // Complex processing logic here
        return data.map(item => ({
            ...item,
            processed: true
        }));
    }
}

// Export class declaration
export class ApiClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    async get(endpoint) {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'GET',
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    }
    
    async post(endpoint, data) {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    }
}

// Class with inheritance
class ExtendedApiClient extends ApiClient {
    constructor(baseURL, apiKey) {
        super(baseURL);
        this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    async authenticatedRequest(endpoint, options = {}) {
        return this.get(endpoint);
    }
}
"""
    
    chunks = chunker.chunk_content(content, Path('test.js'))
    
    print("=== JAVASCRIPT AST CHUNKER TEST ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Content length: {len(content)} characters")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Function: {chunk.metadata.function_name}")
        print(f"  Class: {chunk.metadata.class_name}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  File type: {chunk.metadata.file_type}")
        print(f"  Imports: {len(chunk.metadata.imports)}")
        
        # Show first line of text
        first_line = chunk.text.split('\n')[0].strip()
        print(f"  First line: {first_line}")

if __name__ == '__main__':
    test_javascript_ast()