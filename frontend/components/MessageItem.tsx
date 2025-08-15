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

// STRUCTURED RESPONSE RENDERER COMPONENT
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
  
  // DEFENSIVE RENDERING: Check for renderable structured data
  const hasRenderableStructuredData = message.structuredResponse?.response?.sections && 
    Array.isArray(message.structuredResponse.response.sections) && 
    message.structuredResponse.response.sections.length > 0 &&
    message.structuredResponse.response.sections.some((section: any) => 
      section && typeof section === 'object' && section.content && 
      String(section.content).trim() !== ''
    )
  
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
  
  // DEFENSIVE RENDERING LOGIC
  const renderContent = () => {
    // Show a spinner while the AI is working
    if (message.isProcessing) {
      return (
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-accent" />
          <span>Processing request...</span>
        </div>
      )
    }
    
    // PRIORITY 1: Attempt to render valid structured data if it exists
    if (hasRenderableStructuredData) {
      console.log('🎯 DEFENSIVE: Rendering structured data')
      return (
        <ErrorBoundary fallback={
          <FallbackDisplay message="Structured content rendering failed" />
        }>
          <StructuredResponseRenderer response={message.structuredResponse} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 2: Render clean markdown output from unified pipeline
    if (message.output && typeof message.output === 'string' && message.output.trim() !== '') {
      console.log('🎯 DEFENSIVE: Rendering clean output from unified pipeline')
      return (
        <ErrorBoundary fallback={
          <FallbackDisplay message="Clean output rendering failed" />
        }>
          <MarkdownRenderer content={message.output} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 3 (FALLBACK): Render the raw markdown content if it exists
    if (message.content && message.content.trim() !== '') {
      console.log('🎯 DEFENSIVE: Rendering fallback markdown content')
      return (
        <ErrorBoundary fallback={
          <FallbackDisplay message="Markdown content rendering failed" />
        }>
          <MarkdownRenderer content={message.content} />
        </ErrorBoundary>
      )
    }
    
    // PRIORITY 3 (FINAL FALLBACK): Show explicit message for empty responses
    console.log('🎯 DEFENSIVE: No renderable content found')
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