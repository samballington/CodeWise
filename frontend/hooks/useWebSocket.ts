import { useEffect, useState, useCallback } from 'react'
import { useChatStore, useContextStore, ContextActivity } from '@/lib/store'

export const useWebSocket = () => {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const { addMessage, updateLastMessage, addToolCallToLastMessage } = useChatStore()
  const { setGatheringContext, addContextActivity, clearContextActivities } = useContextStore()

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'
    let ws: WebSocket
    let pingInterval: NodeJS.Timeout

    const connect = () => {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setSocket(ws)
        setConnected(true)
        // Send ping every 25 s to keep the connection alive
        pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 25000)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setConnected(false)
        clearInterval(pingInterval)
        // Try to reconnect after 2 s
        setTimeout(connect, 2000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onmessage = (event: MessageEvent) => {
        try {
          // 🔍 DEBUG: Log ALL WebSocket messages
          console.log('🌐 WEBSOCKET MESSAGE RECEIVED:', {
            messageSize: event.data.length,
            messageType: event.data.substring(0, 100) + '...',
            timestamp: new Date().toISOString()
          })
          
          const data = JSON.parse(event.data)
          
          // 🔍 DEBUG: Log parsed message details
          console.log('🔍 WEBSOCKET PARSED MESSAGE:', {
            type: data.type,
            hasOutput: !!data.output,
            outputLength: data.output?.length || 0,
            hasStructuredResponse: !!data.structured_response,
            hasFormattedResponse: !!data.formatted_response,
            allKeys: Object.keys(data)
          })
          
          if (data.type === 'pong') return // ignore keep-alive
          handleMessage(data)
        } catch (error) {
          console.error('Error parsing message:', error, 'Raw data:', event.data)
        }
      }
    }

    connect()

    return () => {
      clearInterval(pingInterval)
      ws && ws.close()
    }
  }, [])

  const handleMessage = (data: any) => {
    switch (data.type) {
      case 'acknowledgment':
        addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: 'Processing request...',
          timestamp: new Date(),
          isProcessing: true,
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
        // Buffer agent actions - don't display in main content, store for context dropdown
        break

      case 'tool_start':
        // Buffer tool calls - don't display during processing
        const startState = useChatStore.getState()
        const startLastMessage = startState.messages[startState.messages.length - 1]
        if (startLastMessage && startLastMessage.isProcessing) {
          const bufferedToolCalls = startLastMessage.bufferedToolCalls || []
          updateLastMessage({
            bufferedToolCalls: [...bufferedToolCalls, {
              tool: data.tool,
              input: data.input,
              output: '',
              status: 'running'
            }]
          })
        }
        break

      case 'tool_end':
        // Update buffered tool call with output - still don't display
        const endState = useChatStore.getState()
        const endLastMessage = endState.messages[endState.messages.length - 1]
        if (endLastMessage && endLastMessage.bufferedToolCalls && endLastMessage.bufferedToolCalls.length > 0) {
          const updatedBufferedToolCalls = [...endLastMessage.bufferedToolCalls]
          const lastToolCall = updatedBufferedToolCalls[updatedBufferedToolCalls.length - 1]
          updatedBufferedToolCalls[updatedBufferedToolCalls.length - 1] = {
            ...lastToolCall,
            output: data.output,
            status: 'completed'
          }
          updateLastMessage({
            bufferedToolCalls: updatedBufferedToolCalls
          })
        }
        break

      case 'final_result':
        // 🔍 COMPREHENSIVE DEBUG: Log everything about the final result
        console.log('🚨 FINAL_RESULT DEBUG - PARSED DATA:', data)
        console.log('🚨 FINAL_RESULT DEBUG - DATA KEYS:', Object.keys(data))
        console.log('🚨 FINAL_RESULT DEBUG - OUTPUT LENGTH:', data.output?.length || 0)
        console.log('🚨 FINAL_RESULT DEBUG - OUTPUT CONTENT:', data.output)
        console.log('🚨 FINAL_RESULT DEBUG - STRUCTURED_RESPONSE:', data.structured_response)
        console.log('🚨 FINAL_RESULT DEBUG - FORMATTED_RESPONSE:', data.formatted_response)
        console.log('🚨 FINAL_RESULT DEBUG - CONSOLIDATION_METADATA:', data.consolidation_metadata)
        
        // Get the most recent complete context activity to attach to the message
        const contextState = useContextStore.getState()
        const recentCompleteActivity = contextState.recentActivities.find(
          activity => activity.type === 'complete'
        )
        
        console.log('🔍 Final result - context debug:', {
          recentActivities: contextState.recentActivities,
          recentCompleteActivity,
          contextDataToAttach: recentCompleteActivity ? {
            sources: recentCompleteActivity.sources,
            chunksFound: recentCompleteActivity.chunksFound,
            filesAnalyzed: recentCompleteActivity.filesAnalyzed,
            query: recentCompleteActivity.query
          } : undefined
        })
        
        // Update the existing processing message with the final result and move buffered tool calls to visible
        const finalState = useChatStore.getState()
        const finalLastMessage = finalState.messages[finalState.messages.length - 1]
        
        console.log('🚨 FINAL_RESULT DEBUG - CURRENT MESSAGE STATE:', finalLastMessage)
        
        // Clean the output to remove any tool output that might have leaked through
        let cleanOutput = data.output
        if (cleanOutput) {
          cleanOutput = cleanOutput.replace(/✅ Tool output:[\s\S]*?\n\n/g, '')
          cleanOutput = cleanOutput.replace(/🛠️ Running tool:[\s\S]*?\n\n/g, '')
          cleanOutput = cleanOutput.replace(/Unknown function:.*?\n/g, '')
        }
        
        console.log('🚨 FINAL_RESULT DEBUG - CLEAN OUTPUT:', cleanOutput)
        
        const structured = data.structured_response || undefined
        const formatted = data.formatted_response || undefined
        
        console.log('🚨 FINAL_RESULT DEBUG - STRUCTURED:', !!structured)
        console.log('🚨 FINAL_RESULT DEBUG - FORMATTED:', !!formatted)
        
        // Get current structured response to prevent overwriting
        const currentStructured = finalLastMessage?.structuredResponse
        const hasExistingStructured = currentStructured && currentStructured.response && Array.isArray(currentStructured.response.sections)
        
        console.log('🚨 FINAL_RESULT DEBUG - HAS EXISTING STRUCTURED:', hasExistingStructured)
        
        // ALWAYS prioritize structured response over formatted, and never overwrite existing structured
        const finalStructuredResponse = hasExistingStructured ? currentStructured : (structured || formatted || undefined)
        
        console.log('🚨 FINAL_RESULT DEBUG - FINAL STRUCTURED RESPONSE:', finalStructuredResponse)
        
        const messageUpdate = {
          content: (structured || hasExistingStructured) ? '' : cleanOutput,
          isProcessing: false,
          isComplete: true,
          toolCalls: finalLastMessage?.bufferedToolCalls || [],
          bufferedToolCalls: undefined,
          structuredResponse: finalStructuredResponse,
          contextData: recentCompleteActivity ? {
            sources: recentCompleteActivity.sources || [],
            chunksFound: recentCompleteActivity.chunksFound || 0,
            filesAnalyzed: recentCompleteActivity.filesAnalyzed || 0,
            query: recentCompleteActivity.query || ''
          } : undefined
        }
        
        console.log('🚨 FINAL_RESULT DEBUG - MESSAGE UPDATE OBJECT:', messageUpdate)
        
        updateLastMessage(messageUpdate)
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
          // Skip tool output tokens - don't display them in main chat
          if (data.token.includes('Tool output:') || 
              data.token.includes('✅ Tool output') || 
              data.token.includes('🛠️ Running tool:') ||
              data.token.includes('Unknown function:')) {
            break
          }
          
          const state = useChatStore.getState()
          const { messages } = state
          if (messages.length === 0 || messages[messages.length - 1].role !== 'assistant' || messages[messages.length - 1].isComplete) {
            // start new assistant message
            addMessage({
              id: Date.now().toString(),
              role: 'assistant',
              content: data.token,
              timestamp: new Date(),
              isProcessing: false,
            })
          } else {
            // append token to last assistant message
            const lastMsg = messages[messages.length - 1]
            // Only append if not in processing mode
            if (!lastMsg.isProcessing) {
              updateLastMessage({ content: lastMsg.content + data.token })
            } else {
              // Replace processing message with streamed content
              updateLastMessage({ 
                content: data.token,
                isProcessing: false 
              })
            }
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