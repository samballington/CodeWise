'use client'

import React, { useState } from 'react'
import { Message } from '../lib/store'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check, User, Bot, Terminal, FileText, Loader2, CheckCircle, Circle } from 'lucide-react'
import { ContextDropdown } from './ContextDropdown'
import Mermaid from './Mermaid'
import { ErrorBoundary } from './ErrorBoundary'
import FallbackDisplay from './FallbackDisplay'

interface MessageItemProps {
  message: Message
}

interface CopyButtonProps {
  text: string
}

function CopyButton({ text }: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text:', err)
    }
  }

  return (
    <button
      onClick={handleCopy}
      className={`copy-button ${copied ? 'copied' : ''}`}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
    </button>
  )
}

interface ToolOutputProps {
  tool: string
  input: string
  output: string
  status: 'running' | 'completed' | 'error'
}

function ToolOutput({ tool, input, output, status }: ToolOutputProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const getToolIcon = (toolName: string) => {
    switch (toolName.toLowerCase()) {
      case 'terminal':
      case 'run_command':
        return <Terminal className="w-4 h-4" />
      case 'read_file':
      case 'write_file':
        return <FileText className="w-4 h-4" />
      default:
        return <Terminal className="w-4 h-4" />
    }
  }

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-4 h-4 animate-spin text-accent" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <Circle className="w-4 h-4 text-red-500" />
    }
  }

  return (
    <div className="tool-output">
      <div 
        className="tool-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {getToolIcon(tool)}
        <span className="font-medium text-text-primary">{tool}</span>
        <span className="text-text-secondary text-sm">{input}</span>
        <div className="ml-auto flex items-center gap-2">
          {getStatusIcon()}
          <span className="text-xs text-text-secondary">
            {isExpanded ? 'Hide' : 'Show'} output
          </span>
        </div>
      </div>
      
      {isExpanded && (
        <div className="tool-content">
          <div className="relative">
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash"
              PreTag="div"
              customStyle={{
                margin: 0,
                background: '#0f1419',
                border: 'none',
                fontSize: '0.875rem',
              }}
            >
              {output}
            </SyntaxHighlighter>
            <CopyButton text={output} />
          </div>
        </div>
      )}
    </div>
  )
}

// INDIVIDUAL CONTENT BLOCK RENDERERS
function TextBlockRenderer({ block }: { block: any }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert mb-4">
      <ReactMarkdown
        components={{
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            return match ? (
              <div className="relative">
                <SyntaxHighlighter
                  style={vscDarkPlus}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: '0.5rem 0',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                  }}
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
                <CopyButton text={String(children)} />
              </div>
            ) : (
              <code className="bg-background-primary px-1.5 py-0.5 rounded text-sm border border-border" {...props}>
                {children}
              </code>
            )
          }
        }}
      >
        {block.content}
      </ReactMarkdown>
    </div>
  )
}

function ComponentAnalysisRenderer({ block }: { block: any }) {
  return (
    <div className="border border-border rounded-lg p-4 mb-4">
      <h3 className="font-semibold text-lg mb-3 text-text-primary">{block.title}</h3>
      <div className="grid gap-3">
        {block.components.map((component: any, cidx: number) => (
          <div key={cidx} className="border-l-4 border-accent pl-4 hover:bg-background-secondary/30 transition-colors rounded-r-md py-2">
            <div className="font-medium text-base text-text-primary mb-1">{component.name}</div>
            <div className="text-sm text-accent font-mono mb-1">{component.path}</div>
            <div className="text-sm text-text-secondary mb-2">{component.purpose}</div>
            
            {component.key_methods && component.key_methods.length > 0 && (
              <div className="text-xs text-accent mb-1">
                <span className="font-medium">Methods:</span> {component.key_methods.join(', ')}
              </div>
            )}
            
            {component.line_start && component.line_end && (
              <div className="text-xs text-text-secondary">
                <span className="font-medium">Lines:</span> {component.line_start}-{component.line_end}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function CodeSnippetRenderer({ block }: { block: any }) {
  return (
    <div className="border border-border rounded-lg overflow-hidden mb-4">
      <div className="bg-background-secondary px-4 py-2 border-b border-border">
        <h4 className="font-medium text-text-primary">{block.title}</h4>
      </div>
      <div className="relative">
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={block.language || 'text'}
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: 0,
            background: '#0f1419',
            fontSize: '0.875rem'
          }}
        >
          {block.code}
        </SyntaxHighlighter>
        <CopyButton text={block.code} />
      </div>
    </div>
  )
}

function MermaidDiagramRenderer({ block }: { block: any }) {
  return (
    <div className="border border-border rounded-lg p-4 mb-4">
      <h3 className="font-semibold text-lg mb-3 text-text-primary">{block.title}</h3>
      <div className="flex justify-center">
        <Mermaid code={block.mermaid_code} title={block.title} />
      </div>
    </div>
  )
}

function MarkdownTableRenderer({ block }: { block: any }) {
  return (
    <div className="border border-border rounded-lg overflow-hidden mb-4">
      <div className="bg-background-secondary px-4 py-2 border-b border-border">
        <h4 className="font-medium text-text-primary">{block.title}</h4>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-background-primary border-b border-border">
            <tr>
              {block.headers.map((header: string, hidx: number) => (
                <th key={hidx} className="px-4 py-3 text-left font-semibold text-text-primary border-r border-border last:border-r-0">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row: string[], ridx: number) => (
              <tr key={ridx} className="border-b border-border hover:bg-background-secondary/30">
                {row.map((cell: string, cidx: number) => (
                  <td key={cidx} className="px-4 py-3 text-text-primary border-r border-border last:border-r-0 align-top">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// UNIFIED CONTENT BLOCK RENDERER
function UnifiedContentBlockRenderer({ blocks }: { blocks: any[] }) {
  return (
    <div className="space-y-0 w-full">
      {blocks.map((block, idx) => {
        if (!block || typeof block !== 'object' || !block.block_type) {
          console.warn(`Invalid content block at index ${idx}:`, block)
          return (
            <div key={idx} className="text-red-500 text-sm border border-red-500/30 rounded p-2 mb-4">
              Invalid content block: {JSON.stringify(block)}
            </div>
          )
        }

        switch (block.block_type) {
          case 'text':
            return <TextBlockRenderer key={idx} block={block} />
          case 'component_analysis':
            return <ComponentAnalysisRenderer key={idx} block={block} />
          case 'code_snippet':
            return <CodeSnippetRenderer key={idx} block={block} />
          case 'mermaid_diagram':
            return <MermaidDiagramRenderer key={idx} block={block} />
          case 'markdown_table':
            return <MarkdownTableRenderer key={idx} block={block} />
          default:
            console.warn(`Unknown block type: ${block.block_type}`)
            return (
              <div key={idx} className="text-yellow-500 text-sm border border-yellow-500/30 rounded p-2 mb-4">
                Unknown block type: <code>{block.block_type}</code>
                <details className="mt-2">
                  <summary className="cursor-pointer">Block details</summary>
                  <pre className="mt-2 text-xs">{JSON.stringify(block, null, 2)}</pre>
                </details>
              </div>
            )
        }
      })}
    </div>
  )
}

// STRUCTURED RESPONSE RENDERER COMPONENT (LEGACY)
function StructuredResponseRenderer({ response }: { response: any }) {
  const renderTree = (node: any, indent: string = ''): string => {
    try {
      if (!node) return ''
      const label = typeof node.label === 'string' ? node.label : JSON.stringify(node.label)
      let out = `${indent}${label}\n`
      const children = Array.isArray(node.children) ? node.children : []
      for (let i = 0; i < children.length; i++) {
        const isLast = i === children.length - 1
        const branch = isLast ? '└─ ' : '├─ '
        const nextIndent = indent + (isLast ? '   ' : '│  ')
        out += renderTree(children[i], indent + branch)
      }
      return out
    } catch {
      return ''
    }
  }
  
  return (
    <div className="space-y-4 w-full">
      {response.response.sections.map((section: any, idx: number) => {
        switch (section.type) {
          case 'heading':
            const Tag = `h${Math.min(Math.max(section.level || 2, 1), 6)}` as any
            return <Tag key={idx} className="font-semibold mt-2">{section.content}</Tag>
          case 'paragraph':
            return (
              <ReactMarkdown key={idx}>
                {section.content}
              </ReactMarkdown>
            )
          case 'list':
            if (section.style === 'numbered') {
              return (
                <ol key={idx} className="list-decimal pl-6 space-y-1">
                  {(section.items || []).map((it: string, i: number) => <li key={i}>{it}</li>)}
                </ol>
              )
            }
            return (
              <ul key={idx} className="list-disc pl-6 space-y-1">
                {(section.items || []).map((it: string, i: number) => <li key={i}>{it}</li>)}
              </ul>
            )
          case 'diagram':
            if ((section.format || 'mermaid') !== 'mermaid') return null
            const diagramCode = (section.content || '')
              .replace(/;/g, '\n')
              .replace(/\\n/g, '\n')
              .trim()
            return (
              <div key={idx} className="my-2">
                <Mermaid code={diagramCode} title={section.title} />
              </div>
            )
          default:
            return null
        }
      })}
    </div>
  )
}

// MARKDOWN RENDERER COMPONENT  
function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert w-full">
      <ReactMarkdown
        components={{
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            const language = match?.[1] || ''
            
            if (language === 'mermaid') {
              return (
                <div className="my-2">
                  <Mermaid code={String(children).replace(/\n$/, '')} />
                </div>
              )
            }
            
            return match ? (
              <div className="relative">
                <SyntaxHighlighter
                  style={vscDarkPlus}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: '0.5rem 0',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                  }}
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
                <CopyButton text={String(children)} />
              </div>
            ) : (
              <code 
                className="bg-background-primary px-1.5 py-0.5 rounded text-sm border border-border" 
                {...props}
              >
                {children}
              </code>
            )
          },
          table({ children, ...props }: any) {
            return (
              <div className="overflow-x-auto my-4 w-full">
                <table className="w-full table-fixed border-collapse border border-border rounded-lg" {...props}>
                  {children}
                </table>
              </div>
            )
          },
          thead({ children, ...props }: any) {
            return (
              <thead className="bg-background-secondary" {...props}>
                {children}
              </thead>
            )
          },
          tr({ children, ...props }: any) {
            return (
              <tr className="border-b border-border hover:bg-background-secondary/50" {...props}>
                {children}
              </tr>
            )
          },
          th({ children, ...props }: any) {
            return (
              <th className="border border-border px-3 py-2 text-left font-semibold text-text-primary break-words" {...props}>
                {children}
              </th>
            )
          },
          td({ children, ...props }: any) {
            return (
              <td className="border border-border px-3 py-2 text-text-primary align-top break-words whitespace-normal" {...props}>
                {children}
              </td>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user'
  
  // UNIFIED SCHEMA DETECTION: Check for both new and legacy structured data formats
  const hasRenderableStructuredData = 
    // NEW: Detect unified content block format (direct array)
    (message.structuredResponse?.response && 
     Array.isArray(message.structuredResponse.response) &&
     message.structuredResponse.response.length > 0 &&
     message.structuredResponse.response.every((block: any) => block?.block_type)) ||
    // LEGACY: Maintain backward compatibility (nested sections)
    (message.structuredResponse?.response?.sections && 
     Array.isArray(message.structuredResponse.response.sections) && 
     message.structuredResponse.response.sections.length > 0 &&
     message.structuredResponse.response.sections.some((section: any) => 
       section && typeof section === 'object' && section.content && 
       String(section.content).trim() !== ''
     ))
  
  // 🔍 DEBUG: Log message rendering details with defensive checks
  if (!isUser) {
    console.log('🎨 MESSAGEITEM DEFENSIVE DEBUG:', {
      messageId: message.id,
      role: message.role,
      contentLength: message.content?.length || 0,
      outputLength: message.output?.length || 0,
      hasContent: !!(message.content && message.content.trim()),
      hasOutput: !!(message.output && message.output.trim()),
      hasRenderableStructuredData,
      structuredSections: message.structuredResponse?.response?.sections?.length || 0,
      isProcessing: message.isProcessing,
      isError: message.isError,
      canRender: hasRenderableStructuredData || !!(message.output && message.output.trim()) || !!(message.content && message.content.trim())
    })
  }
  
  const renderTree = (node: any, indent: string = ''): string => {
    try {
      if (!node) return ''
      const label = typeof node.label === 'string' ? node.label : JSON.stringify(node.label)
      let out = `${indent}${label}\n`
      const children = Array.isArray(node.children) ? node.children : []
      for (let i = 0; i < children.length; i++) {
        const isLast = i === children.length - 1
        const branch = isLast ? '└─ ' : '├─ '
        const nextIndent = indent + (isLast ? '   ' : '│  ')
        out += renderTree(children[i], indent + branch)
      }
      return out
    } catch {
      return ''
    }
  }
  
  // UNIFIED RENDERING LOGIC
  const renderContent = () => {
    // Show spinner while processing
    if (message.isProcessing) {
      return (
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-accent" />
          <span>Processing request...</span>
        </div>
      )
    }
    
    // PRIORITY 1: Unified content blocks (NEW FORMAT)
    if (message.structuredResponse?.response && Array.isArray(message.structuredResponse.response)) {
      console.log('🎯 UNIFIED: Rendering new content block format')
      return (
        <ErrorBoundary fallback={<FallbackDisplay message="Content block rendering failed" />}>
          <UnifiedContentBlockRenderer blocks={message.structuredResponse.response} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 2: Legacy structured format (BACKWARD COMPATIBILITY)
    if (message.structuredResponse?.response?.sections && 
        Array.isArray(message.structuredResponse.response.sections)) {
      console.log('🎯 LEGACY: Rendering legacy structured format')
      return (
        <ErrorBoundary fallback={<FallbackDisplay message="Legacy structured rendering failed" />}>
          <StructuredResponseRenderer response={message.structuredResponse} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 3: Prevent raw JSON rendering
    if (message.output && typeof message.output === 'string' && message.output.trim() !== '') {
      // Detect and prevent raw JSON from being displayed
      try {
        const parsed = JSON.parse(message.output)
        if (parsed.response && Array.isArray(parsed.response)) {
          console.log('⚠️ BLOCKED: Raw JSON detected in output field')
          return (
            <FallbackDisplay 
              message="Response format error: Raw JSON detected in output field. This indicates a frontend parsing issue." 
            />
          )
        }
      } catch (e) {
        // Not JSON, safe to render as markdown
      }
      
      console.log('🎯 MARKDOWN: Rendering clean output as markdown')
      return (
        <ErrorBoundary fallback={<FallbackDisplay message="Output rendering failed" />}>
          <MarkdownRenderer content={message.output} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 4: Fallback content
    if (message.content && message.content.trim() !== '') {
      console.log('🎯 FALLBACK: Rendering content field as markdown')
      return (
        <ErrorBoundary fallback={<FallbackDisplay message="Content rendering failed" />}>
          <MarkdownRenderer content={message.content} />
        </ErrorBoundary>
      )
    }
    
    // FINAL: Empty response
    console.log('🎯 EMPTY: No renderable content found')
    return <FallbackDisplay message="Assistant returned an empty response." />
  }
  
  // Early return for processing state
  if (message.isProcessing) {
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 message-enter`}>
        <div className={`flex gap-3 ${isUser ? 'max-w-[75%]' : 'max-w-[85%]'} ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser 
              ? 'bg-primary text-white' 
              : 'bg-gradient-to-br from-accent to-primary text-white'
          }`}>
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          </div>
          <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
            <div className="rounded-2xl px-4 py-3 bg-background-secondary text-text-primary border border-border">
              {renderContent()}
            </div>
            <div className="text-xs text-text-secondary mt-2 px-1">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    )
  }
  
  // Legacy helper functions removed - now handled by defensive rendering
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 message-enter w-full`}>
      <div className={`flex gap-3 ${isUser ? 'max-w-[75%]' : 'w-full'} ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-primary text-white' 
            : 'bg-gradient-to-br from-accent to-primary text-white'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>
        
        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} ${isUser ? '' : 'w-full'}`}>
          <div className={`rounded-2xl px-4 py-3 ${isUser ? '' : 'w-full'} ${
            isUser
              ? 'bg-primary text-white'
              : message.isError
              ? 'bg-red-900/50 text-red-200 border border-red-500/50'
              : 'bg-background-secondary text-text-primary border border-border'
          }`}>
            
            {/* DEFENSIVE CONTENT RENDERING */}
            {renderContent()}
            


            
          </div>
          
          {/* Context Dropdown - Only show for assistant messages */}
          {!isUser && (
            <div className="w-full mt-2">
              <ContextDropdown 
                toolCalls={message.toolCalls || []}
                messageId={message.id}
                contextData={message.contextData}
              />
            </div>
          )}
          
          {/* Timestamp */}
          <div className="text-xs text-text-secondary mt-2 px-1">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
} 