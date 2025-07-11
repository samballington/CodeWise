'use client'

import React, { useState, useRef, useEffect } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore, useThemeStore } from '../lib/store'
import MessageList from './MessageList'
import MessageInput from './MessageInput'

export default function ChatInterface() {
  const { sendMessage } = useWebSocket()
  const { messages } = useChatStore()
  const { isDarkMode } = useThemeStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
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
                  <p>"Create a new React component"</p>
                  <p>"Analyze the code in src/main.js"</p>
                  <p>"Run the test suite"</p>
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
        <div className="max-w-4xl mx-auto px-4 py-4">
          <MessageInput onSendMessage={sendMessage} />
        </div>
      </div>
    </div>
  )
} 