'use client'

import React from 'react'
import InteractiveMermaid from './InteractiveMermaid'

const testDiagrams = {
  simple: `graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E`,
    
  complex: `graph TB
    subgraph "Frontend Layer"
        A[React App] --> B[Components]
        B --> C[MessageItem]
        B --> D[ContextDropdown]
        B --> E[InteractiveMermaid]
    end
    
    subgraph "API Layer"
        F[FastAPI Backend] --> G[Agent Router]
        G --> H[Cerebras Agent]
        G --> I[Context Delivery]
        G --> J[Smart Search]
    end
    
    subgraph "Data Layer"
        K[Vector Store] --> L[FAISS Index]
        K --> M[BM25 Index]
        N[File Discovery] --> O[AST Chunker]
        N --> P[Hybrid Search]
    end
    
    subgraph "Processing Pipeline"
        Q[Response Formatter] --> R[Mermaid Validator]
        Q --> S[Table Processor]
        Q --> T[Response Consolidator]
    end
    
    A --> F
    F --> K
    H --> Q
    I --> N
    J --> K
    
    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style K fill:#e8f5e8
    style Q fill:#fff3e0`,
    
  flowchart: `flowchart LR
    Start([Start Process]) --> Input[/User Input/]
    Input --> Validate{Valid Input?}
    Validate -->|No| Error[Show Error]
    Error --> Input
    Validate -->|Yes| Process[Process Data]
    Process --> Transform[Transform Results]
    Transform --> Cache[(Cache Results)]
    Cache --> Display[/Display Output/]
    Display --> End([End Process])
    
    Process --> Log[Log Activity]
    Log --> Audit[(Audit Trail)]
    
    style Start fill:#4caf50
    style End fill:#f44336
    style Process fill:#2196f3
    style Cache fill:#ff9800
    style Audit fill:#9c27b0`
}

export default function MermaidTest() {
  return (
    <div className="p-6 space-y-8 bg-slate-900 min-h-screen">
      <h1 className="text-2xl font-bold text-white mb-6">Interactive Mermaid Test</h1>
      
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-white mb-3">Simple Diagram</h2>
          <InteractiveMermaid 
            code={testDiagrams.simple} 
            title="Simple Decision Flow"
          />
        </div>
        
        <div>
          <h2 className="text-lg font-semibold text-white mb-3">Complex System Architecture</h2>
          <InteractiveMermaid 
            code={testDiagrams.complex} 
            title="CodeWise System Architecture"
          />
        </div>
        
        <div>
          <h2 className="text-lg font-semibold text-white mb-3">Process Flowchart</h2>
          <InteractiveMermaid 
            code={testDiagrams.flowchart} 
            title="Data Processing Pipeline"
          />
        </div>
      </div>
      
      <div className="mt-8 p-4 bg-slate-800 rounded-lg">
        <h3 className="font-medium text-white mb-2">Test Instructions:</h3>
        <ul className="text-slate-300 text-sm space-y-1">
          <li>• Use mouse wheel to zoom in/out</li>
          <li>• Click and drag to pan around</li>
          <li>• Use control buttons for zoom/reset/fullscreen</li>
          <li>• Click help (?) button for keyboard shortcuts</li>
          <li>• Complex diagrams show minimap automatically</li>
          <li>• Try fullscreen mode for detailed viewing</li>
        </ul>
      </div>
    </div>
  )
}