'use client'

import { useEffect, useState } from 'react'
import ChatInterface from '@/components/ChatInterface'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useChatStore } from '@/lib/store'

export default function Home() {
  const { connected, sendMessage } = useWebSocket()
  const { messages } = useChatStore()

  return (
    <main className="flex min-h-screen flex-col">
      <header className="bg-primary text-white p-4 shadow-lg">
        <div className="container mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold">CodeWise</h1>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                connected ? 'bg-green-400' : 'bg-red-400'
              }`}
            />
            <span className="text-sm">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>
      
      <div className="flex-1 container mx-auto p-4">
        <ChatInterface />
      </div>
    </main>
  )
} 