import { useEffect, useState, useCallback } from 'react'
import { useChatStore, useContextStore, ContextActivity } from '@/lib/store'

export const useWebSocket = () => {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const { addMessage, updateLastMessage } = useChatStore()
  const { setGatheringContext, addContextActivity, clearContextActivities } = useContextStore()

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

      case 'context_gathering_start':
        console.log('🔍 Context gathering started:', data.message)
        setGatheringContext(true)
        addContextActivity({
          id: Date.now().toString(),
          type: 'start',
          message: data.message || 'Starting context analysis...',
          timestamp: new Date(),
        })
        break

      case 'context_search':
        console.log('🔎 Context search:', data.source, '-', data.query)
        addContextActivity({
          id: Date.now().toString(),
          type: 'search',
          message: `Grabbing context from: ${data.source}`,
          source: data.source,
          query: data.query,
          timestamp: new Date(),
        })
        break

      case 'context_gathering_complete':
        console.log('✅ Context gathering complete:', data)
        setGatheringContext(false)
        addContextActivity({
          id: Date.now().toString(),
          type: 'complete',
          message: data.no_context 
            ? 'No relevant context found' 
            : `Found ${data.chunks_found || 0} relevant chunks from ${data.files_analyzed || 0} files`,
          sources: data.sources,
          chunksFound: data.chunks_found,
          filesAnalyzed: data.files_analyzed,
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
        // Clear context activities after completion
        setTimeout(() => {
          clearContextActivities()
        }, 2000) // Keep visible for 2 seconds after completion
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
    (messageOrContent: string | { content: string; mentionedProjects: string[] }) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        let content: string
        let mentionedProjects: string[] = []
        
        // Handle both string and object formats
        if (typeof messageOrContent === 'string') {
          content = messageOrContent
        } else {
          content = messageOrContent.content
          mentionedProjects = messageOrContent.mentionedProjects
        }
        
        const message = {
          type: 'user_message',
          content,
          mentionedProjects: mentionedProjects.length > 0 ? mentionedProjects : undefined
        }
        socket.send(JSON.stringify(message))
        
        // Add user message to chat
        addMessage({
          id: Date.now().toString(),
          role: 'user',
          content,
          timestamp: new Date(),
        })
        
        // Clear previous context activities when starting new message
        clearContextActivities()
      }
    },
    [socket, addMessage, clearContextActivities]
  )

  return {
    connected,
    sendMessage,
  }
} 