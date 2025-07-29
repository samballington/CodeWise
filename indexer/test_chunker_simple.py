#!/usr/bin/env python3
"""
Simple test script for the AST chunker
"""

from indexer.ast_chunker import ASTChunker
from pathlib import Path

def test_python_chunking():
    """Test Python AST chunking functionality"""
    chunker = ASTChunker()
    
    # Test content with functions and classes (make it larger to avoid small file chunker)
    content = """import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

def hello_world():
    '''A simple greeting function that prints a greeting message.'''
    print('Hello, World!')
    print('This is a test function')
    return 'greeting'

def another_function(x: int) -> str:
    '''Convert integer to string with validation.'''
    if x < 0:
        raise ValueError('Negative numbers not allowed')
    return str(x)

def complex_function(data: Dict[str, any]) -> Optional[str]:
    '''A more complex function that processes data.'''
    if not data:
        return None
    
    result = []
    for key, value in data.items():
        if isinstance(value, str):
            result.append(f'{key}: {value}')
    
    return ', '.join(result)

@dataclass
class TestClass:
    '''A test class with multiple methods.'''
    name: str
    age: int = 0
    
    def __init__(self, name: str, age: int = 0):
        self.name = name
        self.age = age
    
    def greet(self):
        '''Return a greeting message.'''
        return f'Hello, {self.name}! You are {self.age} years old.'
    
    def update_age(self, new_age: int):
        '''Update the age with validation.'''
        if new_age < 0:
            raise ValueError('Age cannot be negative')
        self.age = new_age

class AnotherClass:
    '''Another test class.'''
    
    def __init__(self):
        self.data = {}
    
    def add_data(self, key: str, value: any):
        '''Add data to the internal dictionary.'''
        self.data[key] = value
    
    def get_data(self, key: str) -> any:
        '''Retrieve data from the internal dictionary.'''
        return self.data.get(key)
"""
    
    chunks = chunker.chunk_content(content, Path('test.py'))
    print(f'Created {len(chunks)} chunks')
    
    for i, chunk in enumerate(chunks):
        print(f'Chunk {i+1}: {chunk.metadata.chunk_type}')
        if chunk.metadata.function_name:
            print(f'  Function: {chunk.metadata.function_name}')
        if chunk.metadata.class_name:
            print(f'  Class: {chunk.metadata.class_name}')
        print(f'  Lines: {chunk.start_line}-{chunk.end_line}')
        print(f'  Imports: {len(chunk.metadata.imports)}')
        print()

def test_javascript_chunking():
    """Test JavaScript chunking functionality"""
    chunker = ASTChunker()
    
    content = """import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';

function HelloComponent(props) {
    const [message, setMessage] = useState('Hello World');
    
    useEffect(() => {
        console.log('Component mounted');
        return () => console.log('Component unmounted');
    }, []);
    
    return <div>{message}</div>;
}

export function AnotherComponent({ title, children }) {
    const [isVisible, setIsVisible] = useState(true);
    
    const handleToggle = () => {
        setIsVisible(!isVisible);
    };
    
    return (
        <div>
            <h1>{title}</h1>
            <button onClick={handleToggle}>Toggle</button>
            {isVisible && children}
        </div>
    );
}

export class MyClass {
    constructor(props) {
        this.props = props;
        this.state = {
            loading: false,
            data: null
        };
    }
    
    async fetchData() {
        this.setState({ loading: true });
        try {
            const response = await axios.get('/api/data');
            this.setState({ data: response.data, loading: false });
        } catch (error) {
            console.error('Error fetching data:', error);
            this.setState({ loading: false });
        }
    }
    
    render() {
        const { loading, data } = this.state;
        
        if (loading) {
            return <div>Loading...</div>;
        }
        
        return (
            <div>
                <h2>Data Component</h2>
                {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : 'No data'}
            </div>
        );
    }
}

class UtilityClass {
    constructor() {
        this.cache = new Map();
    }
    
    getCachedValue(key) {
        return this.cache.get(key);
    }
    
    setCachedValue(key, value) {
        this.cache.set(key, value);
    }
    
    clearCache() {
        this.cache.clear();
    }
}

const arrowFunc = (param1, param2) => {
    console.log('Arrow function with params:', param1, param2);
    return param1 + param2;
};

async function asyncFunction() {
    try {
        const result = await fetch('/api/endpoint');
        const data = await result.json();
        return data;
    } catch (error) {
        console.error('Async function error:', error);
        throw error;
    }
}
"""
    
    chunks = chunker.chunk_content(content, Path('test.js'))
    print(f'Created {len(chunks)} JavaScript chunks')
    
    for i, chunk in enumerate(chunks):
        print(f'Chunk {i+1}: {chunk.metadata.chunk_type}')
        if chunk.metadata.function_name:
            print(f'  Function: {chunk.metadata.function_name}')
        if chunk.metadata.class_name:
            print(f'  Class: {chunk.metadata.class_name}')
        print(f'  Lines: {chunk.start_line}-{chunk.end_line}')
        print()

if __name__ == '__main__':
    print("Testing Python AST Chunker...")
    test_python_chunking()
    
    print("\nTesting JavaScript Chunker...")
    test_javascript_chunking()