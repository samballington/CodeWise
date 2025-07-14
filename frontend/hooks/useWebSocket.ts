import { useEffect, useState, useCallback } from 'react'
import { useChatStore } from '@/lib/store'

export const useWebSocket = () => {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const { addMessage, updateLastMessage } = useChatStore()

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setConnected(true)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setConnected(false)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleMessage(data)
      } catch (error) {
        console.error('Error parsing message:', error)
      }
    }

    setSocket(ws)

    return () => {
      ws.close()
    }
  }, [])

  const handleMessage = (data: any) => {
    switch (data.type) {
      case 'acknowledgment':
        addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
        })
        break

      case 'agent_action':
        updateLastMessage({
          content: `🔧 ${data.action}: ${data.log}`,
        })
        break

      case 'tool_start':
        updateLastMessage({
          content: `🛠️ Running tool: ${data.tool}\nInput: ${data.input}`,
        })
        break

      case 'tool_end':
        updateLastMessage({
          content: `✅ Tool output:\n${data.output}`,
        })
        break

      case 'final_result':
        addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: data.output,
          timestamp: new Date(),
        })
        break

      case 'error':
        addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: `❌ Error: ${data.message}`,
          timestamp: new Date(),
          isError: true,
        })
        break

      case 'completion':
        updateLastMessage({
          isComplete: true,
        })
        break

      case 'stream_token':
        {
          const state = useChatStore.getState()
          const { messages } = state
          if (messages.length === 0 || messages[messages.length - 1].role !== 'assistant' || messages[messages.length - 1].isComplete) {
            // start new assistant message
            addMessage({
              id: Date.now().toString(),
              role: 'assistant',
              content: data.token,
              timestamp: new Date(),
            })
          } else {
            // append token to last assistant message
            const lastMsg = messages[messages.length - 1]
            updateLastMessage({ content: lastMsg.content + data.token })
          }
        }
        break
    }
  }

  const sendMessage = useCallback(
    (content: string) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        const message = {
          type: 'user_message',
          content,
        }
        socket.send(JSON.stringify(message))
        
        // Add user message to chat
        addMessage({
          id: Date.now().toString(),
          role: 'user',
          content,
          timestamp: new Date(),
        })
      }
    },
    [socket, addMessage]
  )

  return {
    connected,
    sendMessage,
  }
} 