'use client'

import { useState, useRef, useEffect } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useChatStore } from '@/lib/store'
import MessageList from './MessageList'
import MessageInput from './MessageInput'

export default function ChatInterface() {
  const { sendMessage } = useWebSocket()
  const { messages } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="flex flex-col h-[calc(100vh-120px)] bg-white rounded-lg shadow-lg">
      <div className="flex-1 overflow-y-auto p-4">
        <MessageList messages={messages} />
        <div ref={messagesEndRef} />
      </div>
      
      <div className="border-t p-4">
        <MessageInput onSendMessage={sendMessage} />
      </div>
    </div>
  )
} 