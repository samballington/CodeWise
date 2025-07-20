'use client'

import React, { useState, useRef, useEffect } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore, useThemeStore } from '../lib/store'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import { ContextPopup } from './ContextPopup'

export default function ChatInterface() {
  const { sendMessage } = useWebSocket()
  const { messages } = useChatStore()
  const { isDarkMode } = useThemeStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Indexer ready state
  const [indexReady, setIndexReady] = useState<boolean>(true)

  // Poll indexer status on mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/indexer/status')
        const json = await res.json()
        setIndexReady(json.ready)
      } catch {
        setIndexReady(true) // fail open
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Enhanced sendMessage handler that supports project mentions
  const handleSendMessage = (message: string, mentionedProjects?: string[]) => {
    // If mentionedProjects are provided, include them in the message context
    const messageWithContext = {
      content: message,
      mentionedProjects: mentionedProjects || []
    }
    
    console.log('Sending message with context:', messageWithContext)
    sendMessage(messageWithContext)
  }

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto relative">
      {/* Context Popup - positioned absolutely for overlay effect */}
      <ContextPopup />
      
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-white font-bold text-xl">CW</span>
              </div>
              <h2 className={`text-2xl font-semibold mb-3 transition-colors duration-200 ${
                isDarkMode ? 'text-text-primary' : 'text-gray-900'
              }`}>
                Welcome to CodeWise
              </h2>
              <p className={`text-lg leading-relaxed transition-colors duration-200 ${
                isDarkMode ? 'text-text-secondary' : 'text-gray-600'
              }`}>
                Your AI development assistant is ready to help. Ask me to read files, write code, run tests, or help with any development task.
              </p>
              <div className={`mt-6 text-sm transition-colors duration-200 ${
                isDarkMode ? 'text-text-secondary' : 'text-gray-600'
              }`}>
                <p className="mb-2">Try asking:</p>
                <div className="space-y-1 text-accent">
                  <p>"@SWE_Project Explain the main functionality"</p>
                  <p>"@workspace How does authentication work?"</p>
                  <p>"Show me the main components in this codebase"</p>
                </div>
                <div className={`mt-4 text-xs px-3 py-2 rounded-lg ${
                  isDarkMode ? 'bg-slate-800 text-slate-300' : 'bg-blue-50 text-blue-700'
                }`}>
                  💡 <strong>Tip:</strong> Use <code className="font-mono">@project</code> mentions to specify which repository you want to analyze
                </div>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input Area */}
      <div className="border-t border-border/30">
        {!indexReady && (
          <div className="p-2 text-center text-sm bg-yellow-100 text-yellow-800">
            🔄 Indexer is still building the code index. Answers may be incomplete for a minute…
          </div>
        )}
        <div className="max-w-4xl mx-auto px-4 py-4">
          <MessageInput onSendMessage={handleSendMessage} disabled={!indexReady} />
        </div>
      </div>
    </div>
  )
} 