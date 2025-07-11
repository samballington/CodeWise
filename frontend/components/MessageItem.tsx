'use client'

import { Message } from '@/lib/store'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface MessageItemProps {
  message: Message
}

export default function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg p-4 ${
          isUser
            ? 'bg-blue-500 text-white'
            : message.isError
            ? 'bg-red-100 text-red-800 border border-red-300'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        <div className="text-sm font-semibold mb-1">
          {isUser ? 'You' : 'CodeWise'}
        </div>
        
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            components={{
              code({ className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '')
                return match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                )
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        
        <div className="text-xs mt-2 opacity-70">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
} 