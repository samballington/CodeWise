'use client'

import React, { useState } from 'react'
import { Message } from '../lib/store'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check, User, Bot, Terminal, FileText, Loader2, CheckCircle, Circle } from 'lucide-react'
import { ContextDropdown } from './ContextDropdown'
import Mermaid from './Mermaid'

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

export default function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user'
  const sr = message.structuredResponse as any | undefined
  
  // 🔍 DEBUG: Log message rendering details
  if (!isUser) {
    console.log('🎨 MESSAGEITEM DEBUG:', {
      messageId: message.id,
      role: message.role,
      contentLength: message.content?.length || 0,
      content: message.content,
      hasStructuredResponse: !!sr,
      structuredResponse: sr,
      isProcessing: message.isProcessing,
      isError: message.isError,
      allMessageKeys: Object.keys(message)
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
  
  // If message is processing, show loading indicator
  if (message.isProcessing) {
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 message-enter`}>
        <div className={`flex gap-3 ${isUser ? 'max-w-[75%]' : 'max-w-[85%]'} ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser 
              ? 'bg-primary text-white' 
              : 'bg-gradient-to-br from-accent to-primary text-white'
          }`}>
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          </div>
          
          {/* Processing Message */}
          <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
            <div className={`rounded-2xl px-4 py-3 bg-background-secondary text-text-primary border border-border`}>
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-accent" />
                <span>Processing request...</span>
              </div>
            </div>
            
            {/* Timestamp */}
            <div className="text-xs text-text-secondary mt-2 px-1">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    )
  }
  
  // Only parse legacy tool outputs for old messages - new messages handle tools via context dropdown
  const parseMessageContent = (content: string): 
    | { type: 'plan'; steps: string[]; remainingContent: string }
    | { type: 'normal'; content: string } => {
    
    // Check for plan/checklist patterns
    const planPattern = /(?:Plan|Steps):\s*\n((?:\d+\.\s*.*\n?)+)/i
    const planMatch = content.match(planPattern)
    
    if (planMatch) {
      const steps = planMatch[1]
        .split('\n')
        .filter(line => line.trim())
        .map(line => line.replace(/^\d+\.\s*/, ''))
      
      return {
        type: 'plan',
        steps,
        remainingContent: content.replace(planMatch[0], '')
      }
    }
    
    return { type: 'normal', content }
  }

  const parsedContent = parseMessageContent(message.content)

  // Enhanced table detection with smart fallbacks
  const processTablesInMarkdown = (md: string): { content: string; hasStructuredData: boolean } => {
    try {
      // First, check if we have structured JSON data
      const jsonBlocks = (md.match(/```json[\s\S]*?```/g) || [])
      if (jsonBlocks.length > 0) {
        const lastJsonBlock = jsonBlocks[jsonBlocks.length - 1]
        const raw = lastJsonBlock.replace(/^```json\n?|```$/g, '')
        try {
          const data = JSON.parse(raw)
          if (data && data.version === 'codewise_structured_v1') {
            // We have structured data - remove JSON blocks from markdown and let structured renderer handle tables
            const cleanedMd = md.replace(/```json[\s\S]*?```/g, '').trim()
            return { content: cleanedMd, hasStructuredData: true }
          }
        } catch {}
      }
      
      // No structured data - apply smart table processing
      const lines = md.split('\n')
      const out: string[] = []
      let i = 0
      
      while (i < lines.length) {
        const line = lines[i]
        const looksLikeTableRow = /\|.*\|/.test(line)
        const next = lines[i + 1] || ''
        const isDivider = /^\s*\|?\s*:?[- ]+:?\s*(\|\s*:?[- ]+:?\s*)+\|?\s*$/.test(next)
        const isValidGfmHeader = looksLikeTableRow && isDivider
        
        // If it looks like a table row but isn't a proper GFM header, try to fix it
        if (looksLikeTableRow && !isValidGfmHeader) {
          const tableBlock: string[] = [line]
          i++
          
          // Collect consecutive table-like lines
          while (i < lines.length && /\|.*\|/.test(lines[i])) {
            tableBlock.push(lines[i])
            i++
          }
          
          // Try to convert to proper GFM table if it looks like tabular data
          if (tableBlock.length >= 2 && canConvertToGfmTable(tableBlock)) {
            const gfmTable = convertToGfmTable(tableBlock)
            out.push(...gfmTable)
          } else {
            // Fall back to code block for ASCII preservation
            out.push('```text')
            out.push(...tableBlock)
            out.push('```')
          }
          continue
        }
        
        out.push(line)
        i++
      }
      
      return { content: out.join('\n'), hasStructuredData: false }
    } catch {
      return { content: md, hasStructuredData: false }
    }
  }
  
  // Helper function to check if table data can be converted to GFM
  const canConvertToGfmTable = (tableLines: string[]): boolean => {
    if (tableLines.length < 2) return false
    
    // Check if all lines have roughly the same number of columns
    const columnCounts = tableLines.map(line => (line.match(/\|/g) || []).length)
    const firstCount = columnCounts[0]
    return columnCounts.every(count => Math.abs(count - firstCount) <= 1)
  }
  
  // Helper function to convert table lines to proper GFM format
  const convertToGfmTable = (tableLines: string[]): string[] => {
    const result: string[] = []
    
    // Add first row as header
    result.push(tableLines[0])
    
    // Create divider based on first row's column count
    const columnCount = (tableLines[0].match(/\|/g) || []).length - 1
    const divider = '| ' + Array(columnCount).fill('---').join(' | ') + ' |'
    result.push(divider)
    
    // Add remaining rows
    result.push(...tableLines.slice(1))
    
    return result
  }

  // Note: We intentionally avoid remark plugins at runtime to prevent preset errors in the browser.
  
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
            
            {/* Plan/Checklist Rendering */}
            {parsedContent.type === 'plan' && (
              <div className="space-y-3">
                <div className="font-medium text-text-primary mb-3">Plan:</div>
                <div className="space-y-2">
                  {parsedContent.steps.map((step, index) => (
                    <div key={index} className="plan-item pending">
                      <Circle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
                {parsedContent.remainingContent && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <ReactMarkdown
                      components={{
                        code({ className, children, ...props }: any) {
                          const match = /language-(\w+)/.exec(className || '')
                          const language = match?.[1] || ''
                          
                          // Handle mermaid code blocks
                          if (language === 'mermaid') {
                            console.log('🎨 ReactMarkdown rendering mermaid code:', children)
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
                            <div className="overflow-x-auto my-4">
                              <table className="w-full border-collapse border border-border rounded-lg" {...props}>
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
                        tbody({ children, ...props }: any) {
                          return (
                            <tbody {...props}>
                              {children}
                            </tbody>
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
                            <th className="border border-border px-4 py-2 text-left font-semibold text-text-primary" {...props}>
                              {children}
                            </th>
                          )
                        },
                        td({ children, ...props }: any) {
                          return (
                            <td className="border border-border px-4 py-2 text-text-primary" {...props}>
                              {children}
                            </td>
                          )
                        },
                      }}
                    >
                      {processTablesInMarkdown(parsedContent.remainingContent).content}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )}
            
            {/* Structured JSON Prompt Rendering */}
            {sr && sr.response && Array.isArray(sr.response.sections) && (
              <div className="space-y-4 w-full">
                {sr.response.sections.map((section: any, idx: number) => {
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
                    case 'table':
                      return (
                        <div key={idx} className="overflow-x-auto my-4 w-full">
                          {section.title && <div className="font-medium mb-2">{section.title}</div>}
                          <table className="w-full table-fixed border-collapse border border-border rounded-lg">
                            <thead className="bg-background-secondary">
                              <tr>
                                {(section.columns || []).map((c: string, i: number) => (
                                  <th key={i} className="border border-border px-3 py-2 text-left font-semibold text-text-primary break-words">{c}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {(section.rows || []).map((row: any[], r: number) => (
                                <tr key={r} className="border-b border-border hover:bg-background-secondary/50">
                                  {row.map((cell: any, c: number) => (
                                    <td key={c} className="border border-border px-3 py-2 text-text-primary align-top break-words whitespace-normal text-sm">{String(cell)}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {section.note && <div className="text-xs text-text-secondary mt-1 italic">{section.note}</div>}
                        </div>
                      )
                    case 'code_block':
                      // Special handling for mermaid code blocks
                      if (section.language === 'mermaid') {
                        console.log('🎨 Converting mermaid code_block to diagram:', section.content)
                        return (
                          <div key={idx} className="my-2">
                            <Mermaid code={section.content || ''} />
                          </div>
                        )
                      }
                      return (
                        <div key={idx} className="relative">
                          <SyntaxHighlighter style={vscDarkPlus} language={section.language || 'text'} PreTag="div" customStyle={{ margin: '0.5rem 0', borderRadius: '8px', fontSize: '0.875rem' }}>
                            {section.content || ''}
                          </SyntaxHighlighter>
                          <CopyButton text={section.content || ''} />
                        </div>
                      )
                    case 'diagram':
                      if ((section.format || 'mermaid') !== 'mermaid') return null
                      // Convert semicolon delimiters to newlines for proper Mermaid formatting
                      const diagramCode = (section.content || '')
                        .replace(/;/g, '\n')  // Convert semicolons to newlines
                        .replace(/\\n/g, '\n') // Handle any escaped newlines
                        .trim()
                      console.log('🎨 Rendering Mermaid diagram:', { 
                        originalContent: section.content,
                        processedCode: diagramCode,
                        hasLineBreaks: diagramCode.includes('\n'),
                        conversionApplied: section.content?.includes(';')
                      })
                      return (
                        <div key={idx} className="my-2">
                          {section.title && <div className="font-medium mb-1">{section.title}</div>}
                          <Mermaid code={diagramCode} />
                        </div>
                      )
                    case 'callout':
                      const tone = section.style || 'info'
                      const toneClasses = tone === 'warning' ? 'bg-yellow-900/30 border-yellow-600/40' : tone === 'error' ? 'bg-red-900/30 border-red-600/40' : tone === 'success' ? 'bg-green-900/30 border-green-600/40' : 'bg-slate-800/40 border-slate-600/40'
                      return (
                        <div key={idx} className={`border rounded px-3 py-2 ${toneClasses}`}>
                          {section.title && <div className="font-medium mb-1">{section.title}</div>}
                          <div>{section.content}</div>
                        </div>
                      )
                    case 'tree':
                      return (
                        <div key={idx}>
                          {section.title && <div className="font-medium mb-2">{section.title}</div>}
                          <pre className="p-3 bg-background-primary border border-border rounded text-sm overflow-x-auto" style={{fontVariantLigatures: 'none'}}>
                            {renderTree(section.root)}
                          </pre>
                        </div>
                      )
                    case 'divider':
                      return <hr key={idx} className="my-4 border-border/50" />
                    case 'quote':
                      return (
                        <blockquote key={idx} className="border-l-4 border-border pl-3 italic">
                          <div>{section.content}</div>
                          {section.attribution && <div className="text-xs text-text-secondary mt-1">— {section.attribution}</div>}
                        </blockquote>
                      )
                    case 'image':
                      return (
                        <div key={idx} className="my-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={section.src} alt={section.alt || ''} className="max-w-full h-auto" />
                          {section.caption && <div className="text-xs text-text-secondary mt-1 text-center">{section.caption}</div>}
                        </div>
                      )
                    default:
                      return null
                  }
                })}
              </div>
            )}

            {/* Fallback: render StandardizedResponse.answer when provided (legacy formatted_response) - ONLY if no structured sections */}
            {(!sr?.response?.sections && sr?.answer) && (
              <div className="prose prose-sm max-w-none dark:prose-invert w-full">
                <ReactMarkdown
                  components={{
                    code({ className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || '')
                      const language = match?.[1] || ''
                      
                      // Handle mermaid code blocks
                      if (language === 'mermaid') {
                        console.log('🎨 Legacy ReactMarkdown rendering mermaid code:', children)
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
                  }}
                >
                  {sr.answer}
                </ReactMarkdown>
              </div>
            )}

            {/* Normal Message Rendering - ONLY if no structured response exists */}
            {!sr && parsedContent.type === 'normal' && (
              <div className="prose prose-sm max-w-none dark:prose-invert w-full">
                <ReactMarkdown
                  components={{
                    code({ className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || '')
                      const language = match?.[1] || ''
                      
                      // Handle mermaid code blocks
                      if (language === 'mermaid') {
                        console.log('🎨 Normal ReactMarkdown rendering mermaid code:', children)
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
                    tbody({ children, ...props }: any) {
                      return (
                        <tbody {...props}>
                          {children}
                        </tbody>
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
                  {(() => {
                    const processed = processTablesInMarkdown(parsedContent.content || message.content)
                    return processed.content
                  })()}
                </ReactMarkdown>
              </div>
            )}
            
            {/* Structured JSON renderer (from model's fenced JSON) */}
            {(() => {
              try {
                const jsonBlocks = (message.content.match(/```json[\s\S]*?```/g) || [])
                if (jsonBlocks.length === 0) return null
                // Use the last JSON block
                const raw = jsonBlocks[jsonBlocks.length - 1].replace(/^```json\n?|```$/g, '')
                const data = JSON.parse(raw)
                if (data && data.version === 'codewise_structured_v1') {
                  return (
                    <div className="mt-4 space-y-6">
                      {(data.tables || []).map((t: any, idx: number) => (
                        <div key={idx}>
                          {t.title && <div className="font-medium mb-3 text-text-primary">{t.title}</div>}
                          <div className="overflow-x-auto my-4 w-full">
                            <table className="w-full table-fixed border-collapse border border-border rounded-lg">
                              <thead className="bg-background-secondary">
                                <tr>
                                  {(t.columns || []).map((c: string, i: number) => (
                                    <th key={i} className="border border-border px-3 py-2 text-left font-semibold text-text-primary break-words">{c}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {(t.rows || []).map((row: any[], r: number) => (
                                  <tr key={r} className="border-b border-border hover:bg-background-secondary/50">
                                    {row.map((cell: any, c: number) => (
                                      <td key={c} className="border border-border px-3 py-2 text-text-primary align-top break-words whitespace-normal text-sm">{String(cell)}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          {t.note && <div className="text-xs text-text-secondary mt-2 italic">{t.note}</div>}
                        </div>
                      ))}
                      {(data.trees || []).map((tree: any, tIdx: number) => (
                        <div key={tIdx}>
                          {tree.title && <div className="font-medium mb-2">{tree.title}</div>}
                          <pre className="p-3 bg-background-primary border border-border rounded text-sm overflow-x-auto" style={{fontVariantLigatures: 'none'}}>
                            {renderTree(tree.root)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )
                }
              } catch {}
              return null
            })()}
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