'use client'

import React, { useState } from 'react'
import { Message } from '../lib/store'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check, User, Bot, Terminal, FileText, Loader2, CheckCircle, Circle } from 'lucide-react'

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
  
  // Parse tool outputs and plan steps from message content
  const parseMessageContent = (content: string): 
    | { type: 'tool_output'; tools: Array<{ tool: string; input: string; output: string; status: 'completed' }> }
    | { type: 'plan'; steps: string[]; remainingContent: string }
    | { type: 'normal'; content: string } => {
    
    // Check for tool output patterns
    const toolPattern = /🛠️ Running tool: (\w+)[\s\S]*?Input: (.*?)[\s\S]*?✅ Tool output:\n([\s\S]*?)(?=🛠️|$)/g
    const toolMatches: RegExpExecArray[] = []
    let match
    while ((match = toolPattern.exec(content)) !== null) {
      toolMatches.push(match)
    }
    
    if (toolMatches.length > 0) {
      return {
        type: 'tool_output',
        tools: toolMatches.map(match => ({
          tool: match[1],
          input: match[2],
          output: match[3],
          status: 'completed' as const
        }))
      }
    }
    
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
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 message-enter`}>
      <div className={`flex gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-primary text-white' 
            : 'bg-gradient-to-br from-accent to-primary text-white'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>
        
        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <div className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-primary text-white'
              : message.isError
              ? 'bg-red-900/50 text-red-200 border border-red-500/50'
              : 'bg-background-secondary text-text-primary border border-border'
          }`}>
            
            {/* Tool Output Rendering */}
            {parsedContent.type === 'tool_output' && (
              <div className="space-y-3">
                {parsedContent.tools.map((tool, index) => (
                  <ToolOutput
                    key={index}
                    tool={tool.tool}
                    input={tool.input}
                    output={tool.output}
                    status={tool.status}
                  />
                ))}
              </div>
            )}
            
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
                      {parsedContent.remainingContent}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )}
            
            {/* Normal Message Rendering */}
            {parsedContent.type === 'normal' && (
              <div className="prose prose-sm max-w-none dark:prose-invert">
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
                  {parsedContent.content || message.content}
                </ReactMarkdown>
              </div>
            )}
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